
class MenuItem:
    def __init__(self, item_name, quantity, size, price, special_instructions=""):
        self.item_name = item_name
        self.quantity = quantity
        self.size = size  # e.g., Small, Medium, Large
        self.price = price
        self.special_instructions = special_instructions

    def calculate_total(self):
        """Calculates the total price for this menu item."""
        return self.price * self.quantity

    def __str__(self):
        """Returns a formatted string representation of the menu item."""
        return (f"{self.quantity} x {self.size} {self.item_name} @ ${self.price:.2f} each"
                + (f" | Special instructions: {self.special_instructions}" if self.special_instructions else ""))


class RestaurantOrder:
    def __init__(self):
        self.order_items = []
        self.delivery = False
        self.delivery_address = ''

    def add_item(self, item_name, quantity, size, price, special_instructions=""):
        """Adds a menu item to the order."""
        item = MenuItem(item_name, quantity, size, price, special_instructions)
        self.order_items.append(item)
        print(f"Added {quantity} x {size} {item_name} to the order.")

    def remove_item(self, item_name):
        """Removes a menu item from the order by name."""
        for item in self.order_items:
            if item.item_name == item_name:
                self.order_items.remove(item)
                print(f"Removed {item_name} from the order.")
                break
        else:
            print(f"{item_name} is not in the order.")

    def calculate_total(self):
        """Calculates the total price of the entire order."""
        return sum(item.calculate_total() for item in self.order_items)

    def show_order(self):
        """Displays the current items in the order."""
        if self.order_items:
            print("Your order contains:")
            for item in self.order_items:
                print(item)
        else:
            print("Your order is empty.")

    def order_summary(self):
        """Returns a formatted string summarizing the order."""
        if not self.order_items:
            return "Your order is empty."

        summary_lines = ["Your order summary:"]
        for item in self.order_items:
            summary_lines.append(str(item))  # Each item string will be added

        total = self.calculate_total()
        summary_lines.append(f"\nTotal: ${total:.2f}")

        # Join the list of lines into a single formatted string
        return "\n".join(summary_lines)


    def apply_discount(self, discount_percentage):
        """Applies a discount to the total order."""
        total = self.calculate_total()
        discount = total * (discount_percentage / 100)
        return total - discount

# Example usage:
# order = RestaurantOrder()
# order.add_item('Burger', 2, 'Large', 8.50, 'No onions')
# order.add_item('Fries', 1, 'Medium', 3.00)
# order.add_item('Soda', 2, 'Small', 1.50, 'Extra ice')
# order.show_order()
#
# total = order.calculate_total()
# print(f"Total: ${total:.2f}")
#
# discounted_total = order.apply_discount(10)
# print(f"Total after 10% discount: ${discounted_total:.2f}")