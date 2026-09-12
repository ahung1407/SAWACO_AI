import os
import sys
import time
import signal
import atexit
import threading
import subprocess
import urllib.request
import urllib.error

try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

_stop_event = threading.Event()
_current_proc = None
_proc_lock = threading.Lock()
_tunnel_url = None
_tunnel_ready = threading.Event()
_supervisor_thread = None
_watchdog_thread = None
_public_ip = "Unavailable"

def get_public_ip(timeout=3):
    """Retrieve public IP address used as the Localtunnel password in web browsers."""
    endpoints = [
        "https://loca.lt/mytunnelpassword",
        "https://api.ipify.org",
        "https://ifconfig.me/ip"
    ]
    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.68.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return resp.read().decode('utf-8').strip()
        except Exception:
            continue
    return "Unavailable (check https://loca.lt/mytunnelpassword)"

def _fetch_ip_worker():
    global _public_ip
    _public_ip = get_public_ip(timeout=3)

def _kill_process_tree(pid):
    """Terminate the process and all child processes (node.exe) on Windows to prevent port/subdomain holding."""
    if not pid:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass

def _tunnel_supervisor(port=8000, subdomain="ai-sawaco"):
    global _current_proc, _tunnel_url
    
    npx_bin = "npx.cmd" if sys.platform == "win32" else "npx"
    expected_url = f"https://{subdomain}.loca.lt"
    
    # Fetch public IP in background so tunnel starts immediately
    threading.Thread(target=_fetch_ip_worker, daemon=True).start()
    
    while not _stop_event.is_set():
        _tunnel_ready.clear()
        _tunnel_url = None
        
        cmd = [npx_bin, "localtunnel", "--port", str(port), "--subdomain", subdomain]
        print(f"[TUNNEL] Requesting fixed subdomain: {expected_url} ...", flush=True)
        
        was_busy = False
        try:
            with _proc_lock:
                if _stop_event.is_set():
                    break
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                _current_proc = proc

            for line in iter(proc.stdout.readline, ''):
                line_str = line.strip()
                if not line_str:
                    continue
                
                if "your url is:" in line_str.lower():
                    assigned_url = line_str.split(":", 1)[1].strip()
                    _tunnel_url = assigned_url
                    
                    if subdomain in assigned_url.lower():
                        _tunnel_ready.set()
                        print("\n" + "=" * 72, flush=True)
                        print("TUNNEL ACTIVE: FIXED SUBDOMAIN ACQUIRED", flush=True)
                        print(f"Public API URL:  {assigned_url}/api/ai/ocr", flush=True)
                        print(f"Swagger Docs:    {assigned_url}/docs", flush=True)
                        print(f"Tunnel Password: {_public_ip} (if opened in browser)", flush=True)
                        print("Backend Header:  Bypass-Tunnel-Reminder: true", flush=True)
                        print("Auto-Recovery:   Active (reconnects automatically on disconnect)", flush=True)
                        print("=" * 72 + "\n", flush=True)
                    else:
                        # Remote server temporarily leased subdomain to previous session
                        was_busy = True
                        print(f"[TUNNEL] Subdomain '{subdomain}' still held by server lease. Re-claiming...", flush=True)
                        break
                        
                elif "tunnel server offline" in line_str.lower() or "connection refused" in line_str.lower():
                    print(f"[TUNNEL] Connection closed by upstream: {line_str}", flush=True)
                    break

            with _proc_lock:
                if _current_proc:
                    _kill_process_tree(_current_proc.pid)
                    _current_proc = None

        except Exception as e:
            if not _stop_event.is_set():
                print(f"[TUNNEL] Process exception: {e}", flush=True)

        if not _stop_event.is_set():
            # Wait 7 seconds if remote server was releasing old lease, otherwise 4 seconds
            wait_sec = 7 if was_busy else 4
            print(f"[TUNNEL] Re-connecting in {wait_sec} seconds to ensure '{subdomain}' acquisition...", flush=True)
            for _ in range(int(wait_sec * 2)):
                if _stop_event.is_set():
                    break
                time.sleep(0.5)

def _watchdog_loop(port=8000, subdomain="ai-sawaco"):
    """
    Heartbeat watchdog: Periodically tests if the public tunnel is responsive.
    Only triggers restart if there are 5 consecutive minutes of total failure
    to avoid false positives caused by temporary network latency spikes.
    """
    time.sleep(30) # Initial grace period
    
    consecutive_failures = 0
    test_url = f"https://{subdomain}.loca.lt/health"
    local_url = f"http://127.0.0.1:{port}/health"
    
    while not _stop_event.is_set():
        if _tunnel_ready.is_set():
            local_alive = False
            try:
                req_loc = urllib.request.Request(local_url)
                with urllib.request.urlopen(req_loc, timeout=3) as resp:
                    if resp.status == 200:
                        local_alive = True
            except Exception:
                local_alive = False

            if local_alive:
                tunnel_alive = False
                try:
                    req_pub = urllib.request.Request(
                        test_url,
                        headers={"Bypass-Tunnel-Reminder": "true", "User-Agent": "HealthWatchdog/1.0"}
                    )
                    with urllib.request.urlopen(req_pub, timeout=12) as resp:
                        if resp.status == 200:
                            tunnel_alive = True
                except Exception:
                    tunnel_alive = False

                if tunnel_alive:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    # Only restart after 4 consecutive failures (over ~4 minutes)
                    if consecutive_failures >= 4:
                        print(f"\n[WATCHDOG] Persistent unresponsiveness detected on '{subdomain}.loca.lt'.", flush=True)
                        print("[WATCHDOG] Restarting tunnel connection...", flush=True)
                        with _proc_lock:
                            if _current_proc:
                                _kill_process_tree(_current_proc.pid)
                        consecutive_failures = 0

        for _ in range(60): # Check once every 60 seconds
            if _stop_event.is_set():
                break
            time.sleep(1)

def start_tunnel_manager(port=8000, subdomain="ai-sawaco"):
    """Start Tunnel Supervisor and Watchdog in background daemon threads."""
    global _supervisor_thread, _watchdog_thread
    _stop_event.clear()
    
    _supervisor_thread = threading.Thread(
        target=_tunnel_supervisor,
        args=(port, subdomain),
        daemon=True,
        name="TunnelSupervisorThread"
    )
    _supervisor_thread.start()
    
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop,
        args=(port, subdomain),
        daemon=True,
        name="TunnelWatchdogThread"
    )
    _watchdog_thread.start()

def stop_tunnel_manager():
    """Stop the tunnel and clean up all child processes."""
    global _current_proc
    _stop_event.set()
    _tunnel_ready.clear()
    
    with _proc_lock:
        if _current_proc:
            print("[TUNNEL] Terminating tunnel process...", flush=True)
            _kill_process_tree(_current_proc.pid)
            _current_proc = None

atexit.register(stop_tunnel_manager)

if __name__ == "__main__":
    print("Starting standalone Tunnel Manager...", flush=True)
    try:
        start_tunnel_manager(port=8000, subdomain="ai-sawaco")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        stop_tunnel_manager()
