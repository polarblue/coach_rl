# coach_rl a thought experiment
* modifying the target of a model based on a pretrained policy 

## walkthrough

1) creates a simple dqn policy:
   * trains it to achieve an optimal behavioral policy*
   * trains an lstm to predict a modifying value(coach)*
 
 
2) instanciate a new dqn agent in same enviroment:
    * train agent while modifying target value with the lstm
  
 Results :
   * caps max reward because any value after 200 for modifier is equal to 1
   * while losing at 200 or above modifies the value by 10*

  
