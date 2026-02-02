This repo provides a menu usable by an multi agent system conversing with customers and taking orders at mcdonald's drive thrus at different locations.

The menu for a particular restaurant location will be stored in json. once the menu loads into memory all agents will have full access to what is available for sale at that particular restaurant. Different restaurant locations will have different menus, even if it is for the same restaurant brand like mcdonalds 

Ordering. 
conversation agent will first greet the customer. "Welcome to McDonald's how can I help you today?"

The customer will respond with one item at a time, e.g. I want a large fries. The customer might also respond with a quantity greater than one of an single item, e.g. I want 4 cheeseburgers.   

The conversation agent will then pass the order via text to the order system for potential inclusion in the customer's order. The order system will respond with Success or Failure for each item, and convey this information to the customer.  This customer order - conversation agent response cycle will continue in a loop until the customer signifies that they are done ordering. then, the conversation agent will read back the Item.name and Item.quantity for each item in the order, and then thank the customer for choosing McDonalds.

The order agent orchestrates several responsibilities, each of which is its own sub agent.


First, the validation sub agent should validate the item the customer orders against the menu. It should use attempt strict matching and fairly strict fuzzy matching against the Item.name for the items on the menu. Use the returns python library to return a valid Success(Item) with the appropriate item name and quantity and other fields of the Pydantic Item model. Use the returns python library. it should return Failure("Item Not Found") if it cannot match. for example, if the customer Failure since tacos are not on the menu given, even though tacos are a popular thing that people might order from a drive thru. The menu is the final arbiter of what the customer can and cannot add to their order. If it is not on the menu, must return Failure("Item not found.")


If the validation agent returns success, then next phase is to add the Item to the customer's current order. 

Each customer will get assigned a new Order object with a 
determinthe as the customer orders items,the agent will use its order tool to validate that the item is available for sale, and then ensure it has the correct item details that the client wants.