This repo provides a menu usable by an workflow with structured inputs and outputs being passed back and forth to LLMs.
The workflow covers taking customer order and providing the customer's final order as a drive thru attendant taking orders at mcdonald's drive thrus at different locations.

The menu for a particular restaurant location will be stored in json. once the menu loads into memory all agents will have full access to what is available for sale at that particular restaurant. Different restaurant locations will have different menus, even if it is for the same restaurant brand like mcdonalds 

Ordering. 
workflow will first greet the customer. as a helpful employee of the restaurant "Welcome to McDonald's how can I help you today?"

The customer will respond with one item at a time, e.g. I want a large fries. The customer might also respond with a quantity greater than one of an single item, e.g. I want 4 cheeseburgers.   

The workflow will then pass on an Item object to the order system for potential inclusion in the customer's order. The order system will validate the Item. and  will respond with Success or Failure, and convey this information to the customer, and save the order to date in an Order object. 

 This customer orders --> order system responds  part of the workflow  will continue in a loop until the customer signifies that they are done ordering. 
 
 then, the conversation agent will read back the Item.name and Item.quantity for each item in the Order, and then thank the customer for choosing McDonalds.

The order workflow has several responsibilities

First, the Item the customer requests should be validated against the menu. It should use attempt strict matching and fairly strict fuzzy matching against the Item.name and Item.quantity for the items on the menu. return a Success and add to state if the item is valid,  and it should return Failure if it Item.name does not match or the Item.quantity is invalid cannot match.

for example, if the customer Item.name = "tacos" and Item.quantity = 5, This should be a failure since tacos are not on the menu given.  even though tacos are a popular thing that people might order from a drive thru, if the menu does not have taco or tacos on it, it is an invalid Item.  The menu is the final arbiter of what the customer can and cannot add to their order. If it is not on the menu, must return Failure


If the validation is successful, then next phase is to add the Item to the customer's current Order. 

the Item should get appended to Order.items. 

Each customer will get assigned a new Order object with the as the customer orders items,the agent will use its order tool to validate that the item is available for sale, and then ensure it has the correct item details that the client wants.