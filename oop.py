# ==============================================
# Python OOP: Everything Explained with Examples
# ==============================================

# --------------------------
# 1. Classes and Objects
# --------------------------
class Dog:
    """A simple Dog class to demonstrate basic OOP concepts."""

    def __init__(self, name):
        """Constructor method to initialize instance attributes."""
        self.name = name  # Instance attribute

    def bark(self):
        """Instance method to demonstrate behavior."""
        print(f"{self.name} says woof!")


# Create an object (instance of the Dog class)
my_dog = Dog("Buddy")
my_dog.bark()  # Output: "Buddy says woof!"


# --------------------------
# 2. Special Methods (Dunder Methods)
# --------------------------
class Vector:
    """A Vector class to demonstrate operator overloading using dunder (__method name__) methods."""

    def __init__(self, x, y):
        """Constructor to initialize x and y coordinates."""
        self.x, self.y = x, y

    def __add__(self, other):
        """Overload the + operator."""
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        """Provide a string representation for debugging."""
        return f"Vector({self.x}, {self.y})"


v1 = Vector(2, 3)
v2 = Vector(1, 4)
print(v1 + v2)  # Output: Vector(3, 7)


# --------------------------
# 3. Inheritance
# --------------------------
class Animal:
    """Base class for animals."""

    def __init__(self, name):
        """Constructor to initialize the name."""
        self.name = name

    def speak(self):
        """Abstract method to be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement")


class Cat(Animal):
    """Cat class inheriting from Animal."""

    def speak(self):
        """Override the speak method."""
        return "Meow!"


class Dog(Animal):
    """Dog class inheriting from Animal."""

    def speak(self):
        """Override the speak method."""
        return "Woof!"


# Polymorphism in action
def animal_sound(animal):
    """Function to demonstrate polymorphism."""
    print(animal.speak())


animal_sound(Cat("Whiskers"))  # Output: "Meow!"
animal_sound(Dog("Buddy"))  # Output: "Woof!"


# --------------------------
# 4. Encapsulation
# --------------------------
class Person:
    def __init__(self, name, age):
        self.name = name  # Public attribute
        self._age = age   # Protected attribute (convention, not enforced)
        self.__id = 1234  # Private attribute (name mangling applied)

    def display(self):
        print(f"Name: {self.name}, Age: {self._age}, ID: {self.__id}")

# Create an object of the Person class
p = Person("Ali", 30)

# Accessing public attribute
print(p.name)  # Output: Ali (Accessing public attribute is allowed)

# Accessing protected attribute (not recommended, but possible)
print(p._age)  # Output: 30 (Accessing protected attribute is allowed, but it's a convention to avoid it)

# Accessing private attribute (will cause an error)
# print(p.__id)  # Error: AttributeError (Private attribute cannot be accessed directly)

# Using a method to display all attributes
p.display()  # Output: Name: Ali, Age: 30, ID: 1234


class Circle:
    """Circle class to demonstrate encapsulation and properties."""

    def __init__(self, radius):
        """Constructor to initialize the radius."""
        self._radius = radius  # Protected attribute

    @property
    def radius(self):
        """Getter for radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Setter for radius with validation."""
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius cannot be negative")


c = Circle(5)
c.radius = 10  # Uses setter
print(c.radius)  # Output: 10


# --------------------------
# 5. Class and Static Methods
# --------------------------
class Date:
    """Date class to demonstrate class and static methods."""

    def __init__(self, year, month, day):
        """Constructor to initialize date."""
        self.year, self.month, self.day = year, month, day

    @classmethod
    def from_string(cls, date_str):
        """Alternative constructor using a class method."""
        y, m, d = map(int, date_str.split('-'))
        return cls(y, m, d)

    @staticmethod
    def is_valid(date_str):
        """Static method to validate a date string."""
        try:
            y, m, d = map(int, date_str.split('-'))
            return m <= 12 and d <= 31
        except:
            return False


date = Date.from_string("2023-10-23")
print(Date.is_valid("2023-13-32"))  # Output: False


# --------------------------
# 6. Advanced Topics
# --------------------------

# Slots for Memory Optimization
class Person:
    """Person class to demonstrate __slots__ for memory optimization."""
    __slots__ = ['name', 'age']  # Restricts instance attributes

    def __init__(self, name, age):
        """Constructor to initialize name and age."""
        self.name, self.age = name, age


# Composition Over Inheritance
class Engine:
    """Engine class to demonstrate composition."""

    def start(self):
        """Start the engine."""
        print("Engine started")


class Car:
    """Car class composed of an Engine."""

    def __init__(self):
        """Constructor to initialize the Engine."""
        self.engine = Engine()

    def start(self):
        """Start the car by starting the engine."""
        self.engine.start()


my_car = Car()
my_car.start()  # Output: "Engine started"


# Mixins for Adding Functionality
class JSONMixin:
    """Mixin to add JSON serialization functionality."""

    def to_json(self):
        """Convert the object to a JSON string."""
        import json
        return json.dumps(self.__dict__)


class User(JSONMixin):
    """User class using the JSONMixin."""

    def __init__(self, name):
        """Constructor to initialize the name."""
        self.name = name


u = User("Alice")
print(u.to_json())  # Output: {"name": "Alice"}


# --------------------------
# 7. Best Practices and Tricks
# --------------------------

# Avoid Mutable Class Attributes
class Dog:
    """Dog class to demonstrate avoiding mutable class attributes."""

    def __init__(self, name):
        """Constructor to initialize name and tricks."""
        self.name = name
        self.tricks = []  # Instance attribute (safer than class attribute)


# Custom Exceptions
class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


# --------------------------
# 8. Design Patterns (Brief)
# --------------------------

# Singleton Pattern
class Singleton:
    """Singleton class to ensure only one instance exists."""

    _instance = None

    def __new__(cls):
        """Override __new__ to control instance creation."""
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


# Factory Pattern
class Button:
    """Base Button class."""

    def render(self):
        """Render the button."""
        pass


class SubmitButton(Button):
    """Submit button implementation."""

    def render(self):
        """Render the submit button."""
        print("Render Submit Button")


class ResetButton(Button):
    """Reset button implementation."""

    def render(self):
        """Render the reset button."""
        print("Render Reset Button")


class ButtonFactory:
    """Factory class to create buttons."""

    def create_button(self, type):
        """Create a button based on the type."""
        if type == "submit":
            return SubmitButton()
        elif type == "reset":
            return ResetButton()
        else:
            raise ValueError("Invalid button type")


factory = ButtonFactory()
submit_button = factory.create_button("submit")
submit_button.render()  # Output: "Render Submit Button"


# ==============================================
# End of Python OOP Examples
# ==============================================