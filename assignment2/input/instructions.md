# Assignment: DSOM 2026-2

**Topic:** CTR Prediction

**Hand in:** Report (PDF) and Python Notebook (link) in Canvas

**Deadline:** May 13, 2026

# Context and scope

Avazu is a global mobile advertising platform. It links advertisers and publishers. CTR prediction is essential for them. Your team has to advice on implementing a CTR prediction model.

They have provided a free for use dataset on Kaggle. Go to https://www.kaggle.com/c/avazu-ctr-prediction and download the train dataset (the "train.gz", not the smaller "test.gz"). This dataset contains approximately 40 million impressions over a 10 day period, ordered chronologically. Each impression has a click/no-click indicator and a number of attributes (such as website and time). All attributes should be interpreted as nominal categorical variables (even if the values are integers). Because of the size of the dataset, I would advise NOT to load the entire dataset in memory at once.

The objective of the CTR prediction model is to predict the probability that an impression will turn into a click. Apply both the Logistic Regression model using Stochastic Gradient Descent and the Naïve Bayes model.

Consider which features to incorporate into your models.

Train the models using online learning (observation by observation or in mini batches).

Evaluate the models on unseen data using the log loss metric in an efficient way.

# Report

Hand in a short report (next to your code/notebook) where you discuss your observations and advice for implementing the CTR prediction model:

- What are your insights from exploring the dataset.
- How you selected which variables to include in your models.
- How you chose the parameters of your models.
- How you evaluated the models on unseen data.
- How the calculation times of the two models compare.
- The practical difficulties you encountered while implementing your CTR prediction models on a real-life dataset.
- Which model you would advise to implement.

Always explain clearly what you do and provide clear arguments for the choices your make (such as the parameter settings you choose).