# Rule
1. Theory 30 % 
2. Problem solving and Case Studies 70 %
> keeping in mind this ratio may flexes depending on the topic

**Theory :** Notes 
**Problem Solving :** `DeepML` ( for Now )
**Case Study :** GitHub Repos, Kaggle Competition, Blogs, X 

# Simple Funda
1. Go in sequence without being overwhelmed by the curriculum
2. try to have a logic journal
	1. Your approach and confessions
	2. after looking at solution, where you stuck and what was the reason, specific notes
	3. Links to past logical issues
	4. last: keep it brutally factual ( even if it hurts our ego )
3. Have a GitHub Repo and push everything there
4. Participate in ml competition to see where you stand , keep an eye on X for any challenges or opportunity
5. DeepML platform will help you implement theory to code but not the application of theory in use cases, for that you have to work on real problems
6. as ML and DL have larger datasets to work on they may not fit in our laptop or any consumer grade devices so practice of using cloud notebooks like colab, lightning ai.
7. Thing can be ambiguous and messy so always care about the evaluation metrics ( multiple ) to validate you are on right track or not
	1. Also remember: people lie with metrics all the time. So learn which metrics fail, and when.
8. you will get humbled multiple times for a simple concept not hitting you at the time of logic building but it's the process
9. Real application skills come from building something end-to-end and watching it break in embarrassing ways.


# Few advanced Funda
1. Seed everything. Version notebooks. Log experiments. This habit alone separates people who “play with ML” from people who _engineer_ ML.
2. Start by reading just the abstract, method diagram, and one equation you understand. Build the muscle. Don’t outsource thinking to blogs forever. ( far later in our curriculum )
3. in the end build at least 2 real projects 
	1. One with classical ML
	2. One Model DL
	as these teach completely different instincts
4. Learn debugging. Seriously.
	1. bad preprocessing
	2. bad splits
	3. train - test leakage
	4. cursed batch norms
	5. exploding gradients
5. Learn tooling like an adult.
	1. pytest ( writing test cases in order to not check thing manually )
	2. logging
	3. Hydra and config management
	4. MLflow and W&B ( there are for visualization of progress of training epochs )
	As ML isn’t just models; it’s plumbing.
6. Learn tradeoffs, not tricks.
	You should be able to explain:
	- Why you’d choose XGBoost instead of a neural network.
	- Why a simple baseline is sometimes unbeatable.
	- Why inference cost ruins half the fancy architectures you see online.
7. **Collaborate early.**  
	ML alone turns into tunnel vision fast. Code reviews from smarter people hurt, but they fix you.
8. after a certain amount of time you have to move out of colab and consumer grade device inference, instead you have to rent GPU on  a proper cloud VM and set it up like a real engineer. it's a senior level skill


# Few Code Funda
1. Don’t trust your code. Make it prove itself.
	1. A model training script that “seems to work” is usually lying.
	2. as bugs ( silent ones ) kill accuracy more than "bad models"
2. Wrap everything into function
	- like modular blocks
3. Always start with a dumb baseline
	1. our "fancy model" ( that we assumed to work ) should beat the baseline models
4. Log everything 
5. Fix random seeds
6. always create a config to make things reproducible
7. Use device-agnostic code from day one. ( Like CPU or GPU )
8. Don’t skip gradient checks.
	1. Half the “model not learning” problems are due to frozen gradients or bad loss.
	2. so always check which are not going to be trained and which are going to be trained
9. Always use validation set with train and test
10. Learn vectorization early. ( instead of loop use the vectorization properties of libraries like numpy and pandas )
11. Save Models
12. Never trust default settings blindly
13. Don't ignore warnings
14. Start every project with a minimal working toy.
	1. Personally i faced a lot as there is a fact "If your architecture fails on a tiny dataset, it will absolutely fail on a big one."
15. Name your files appropriately like data/, src/, models/, notebooks/, ...
16. always use type hints
17. print parameter counts 
18. Unit test your data pipeline.
	- Data bugs are more common than model bugs.
19. For hyperparameters, search > intuition.
	1. we shouldn't guess instead use gridsearch, random search
20. Comment _why_, not what.
