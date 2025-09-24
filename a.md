Dưới đây là đề của 35 câu hỏi bạn đã cung cấp:

**Question 1. (Java basic)**
What is the default FetchType of @OneToMany mapping in JPA?
*   LAZY
*   EAGER
*   NONE

**Question 2. (Java basic)**
What is the purpose of JPA's PersistenceContext?
*   To store managed entities and their state
*   To manage database connections
*   To define database schemas
*   To execute database queries

**Question 3. (Java basic)**
Which annotation is used to map an enumerated type property in JPA?
*   @Enumerated
*   @Enum
*   @EnumType
*   @Type

**Question 4. (Java basic)**
What is the purpose of the @JoinColumn annotation in JPA?
*   To join multiple tables in the query
*   To specify the foreign key column in a relationship
*   To define a primary key column
*   To create an index on a column

**Question 5. (Java basic)**
When a field is declared static, there will be:
*   a copy of the field in each class object
*   only one copy of the field in memory
*   a copy of the field for each static method in the class
*   only two copies of the field in memory

**Question 6. (Java basic)**
Assume the class BankAccount has been created, and the following statement correctly creates an instance of the class:
`BankAccount account = new BankAccount(5000.0);`
What is TRUE about the following statement?
`System.out.println(account);`
*   The method will display unreadable binary data on the screen.
*   A compiler error will occur.
*   The account object's toString method will be implicitly called.
*   A runtime error will occur.

**Question 7. (Java basic)**
This is a group of related classes
*   archive
*   package
*   collection
*   attachment

**Question 8. (Java basic)**
Overloading means multiple methods in the same class:
*   have the same name, but different return types
*   have different names, but the same parameter list
*   have the same name, but different parameter lists
*   perform the same function

**Question 9. (Java basic)**
When an object is passed as an argument to a method, what is passed into the method's parameter variable?
*   the class name
*   the object's memory address
*   the values for each field
*   the method names

**Question 10. (Java 8)**
Stream mutable reduction operation \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ creates a new collection of elements containing the results of the stream's prior operations.
*   combine
*   accumulate
*   gather
*   collect

**Question 11. (Java 8)**
Stream reduction operation \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ uses the elements of a collection to produce a single value using an associative accumulation function (e.g., a lambda that adds two elements).
*   reduce
*   condense
*   combine
*   associate

**Question 12. (Java 8)**
Intermediate Stream operation \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ results in a stream with the specified number of elements from the beginning of the original stream.
*   distinct
*   map
*   filter
*   limit

**Question 13. (Java 8)**
The intermediate Stream operation \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ results in a stream containing only the unique elements.
*   distinct
*   map
*   filter
*   limit

**Question 14. (Java 8)**
The intermediate Stream operation \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ results in a stream containing only the elements that satisfy a condition.
*   distinct
*   filter
*   map
*   limit

**Question 15. (Java 8)**
Intermediate operations are \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_; they aren't performed until a terminal operation is invoked. This allows library developers to optimize stream-processing performance.
*   eager
*   idle
*   lazy
*   inactive

**Question 16. (Java 8)**
Which of the following statements is false?
*   A lambda that receives two ints, x and y, and returns their sum is (int x, int y) -> {return x + y;}
*   A lambda's parameter types may be omitted, as in: (x, y) -> {return x + y;} in which case, the parameter and return types are set to the lambda's default type.
*   A lambda with a one-expression body can be written as: (x, y) -> x + y In this case, the expression's value is implicitly returned.
*   When a lambda's parameter list contains only one parameter, the parentheses may be omitted, as in: value -> System.out.printf("%d ", value)

**Question 17. (Java 8)**
What is the meaning of () in the following lambda?
`() -> System.out.println("Welcome to lambdas!")`
*   The lambdas parameters are inferred
*   The lambda has an empty parameter list
*   The lambdas parameters are supplied by a method reference
*   The given expression is not a valid lambda

**Question 18. (Java 8)**
A lambda expression represents a(n) \_\_\_\_\_\_\_\_\_\_\_\_\_\_ method—a shorthand notation for implementing a functional interface.
*   functional
*   unnamed
*   undesignated
*   anonymous

**Question 19. (Java 8)**
The basic generic functional interface \_\_\_\_\_\_\_\_\_\_\_\_\_\_ in package `java.util.function` contains a method `test` that takes a T argument and returns a boolean. Tests whether the T argument satisfies a condition.
*   Consumer
*   Function
*   Supplier
*   Predicate

**Question 20. (Java)**
What will be printed to the console output?
```java
public static void main(String[] args) {
    Collection<?> collection =
        new HashSet<>(
            new ArrayList<>(new LinkedHashSet<>(
                Arrays.asList(3, 2, 3, 1, 7))));
    System.out.println(collection);
}
```
*  
*  
*  
*   []
*  

**Question 21. (Java)**
What will be printed to the console output?
```java
public static void main(String[] args) {
    Predicate<Integer> predicate = var -> var % 7 == 0;
    Function<Integer, Integer> transformer = var -> var + 2017;
    Stream.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        .filter(predicate)
        .map(transformer)
        .map(integer -> String.valueOf(integer + 1))
        .findAny()
        .ifPresent(System.out::println);
}
```
*   2017
*   Nothing will be printed
*   2025
*   7
*   2018

**Question 22. (Java)**
Consider the following program:
```java
import java.util.*;
class Diamond {
    public static void main(String[] args) {
        List list1 = new ArrayList<>(Arrays.asList(1, "two", 3.0)); // ONE
        List list2 = new LinkedList<>
            (Arrays.asList(new Integer(1), new Float(2.0F), new Double(3.0))); // TWO
        list1 = list2; // THREE
        for(Object element : list1) {
            System.out.print(element + " ");
        }
    }
}
```
Which one of the following describes the expected behavior of this program?
*   The program results in compiler error in line marked with comment ONE.
*   The program results in compiler error in line marked with comment TWO.
*   The program results in compiler error in line marked with comment THREE.
*   When executed, the program prints 1 2.0 3.0.
*   When executed, this program throws a ClassCastException.

**Question 23. (Java)**
Consider the following program:
```java
abstract class AbstractBook {
    public String name;
}

interface Sleepy {
    public String name = "undefined";
}

class Book extends AbstractBook implements Sleepy {
    public Book(String name) {
        this.name = name; // LINE A
    }
    public static void main(String[] args) {
        AbstractBook philosophyBook = new Book("Principia Mathematica");
        System.out.println("The name of the book is " + philosophyBook.name); // LINE B
    }
}
```
Which one of the following options correctly describes the behavior of this program?
*   The program will print the output "The name of the book is Principia Mathematica".
*   The program will print the output "The name of the book is undefined".
*   The program will not compile and result in a compiler error "ambiguous reference to name" in line marked with comment LINE A.
*   The program will not compile and result in a compiler error "ambiguous reference to name" in line marked with comment LINE B.

**Question 25. (General)**
Which statement is described in following flowchart? (Flowchart shows "Body of Loop" then "Test expression", if true, goes back to "Body of Loop", if false, goes to "Statement just below Loop")
*   for loop
*   while loop
*   do while
*   if
*   switch statement

**Question 26. (General)**
What does the following function do? (Assume that Swap function will switch the contents of two elements; n is the length of array a)
```c++
void Fun(int a[], int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (a[i] < a[j])
                Swap(a[i], a[j]);
        }
    }
}
```
*   It generates an runtime error.
*   It sorts the elements of array a in ascending order, according to the value.
*   It sorts the elements of array a in descending order, according to the value.
*   It does nothing.

**Question 27. (General)**
What value does function mystery return when called with a value of 4?
```c++
int mystery ( int number )
{
    if ( number <= 1 )
        return 1;
    else
        return number * mystery( number - 1 );
}
```
*   1
*   4
*   12
*   24

**Question 28. (General)**
Which is list of characteristics describes best the object oriented programming?
*   Class, object, interface
*   Data driven, layered architecture, persistence
*   Inheritance, encapsulation, polymorphism
*   Variable, method, object

**Question 29. (General)**
Look at the following code. Which line will cause a compiler error?
```java
Line 1  public class ClassA {
Line 2      {
Line 3          public ClassA() {}
Line 4          public final int method1 (int a){}
Line 5          public double method2(int b){}
Line 6      }
Line 7  public class ClassB extends ClassA
Line 8      {
Line 9          public ClassB(){}
Line 10         public int method1(int b){}
Line 11         public double method2(double c){}
Line 12     }
```
*   4
*   5
*   10
*   11

**Question 30. (General)**
In OOP concept, Which of the following class members should usually be private?
*   Methods
*   Constructors
*   Variables (or fields)
*   All of the above

**Question 31. (General)**
How many Book objects are created by the following statement?
`Book[] books = new Book[10];`
*   10
*   0
*   5
*   None of the above

**Question 32. (General)**
Using a binary search, what is the maximum number of comparisons required to find a search key in a 31-element sorted array?
*   4
*   5
*   6
*   31
*   32

**Question 33. (General)**
Static class variables:
*   are final.
*   are public.
*   are private.
*   are shared by all objects of a class.

**Question 34. (General)**
Consider the code segment below.
```c++
if (gender == 1) {
    if (age >= 65)
        ++seniorFemales;
}
```
This segment is equivalent to which of the following?
a. `if (gender == 1 || age >= 65) ++seniorFemales;`
b. `if (gender == 1 && age >= 65) ++seniorFemales;`
c. `if (gender == 1 AND age > 65) ++seniorFemales;`
d. `if (gender == 1 OR age >= 65) ++seniorFemales;`
*   a
*   b
*   c
*   d

**Question 35. (General)**
In general, linked lists allow:
*   Insertions and removals anywhere.
*   Insertions and removals only at one end.
*   Insertions at the back and removals from the front.
*   None of the above.

**Bonus Question (from an image without a number):**
What is the result if we call `mystery(8, 2)`?
```c++
long mystery(long n, long k) {
    long something = n;
    long count = -1;
    while (something >= 1) {
        something /= k;
        count += 1;
    }
    return count;
}
```