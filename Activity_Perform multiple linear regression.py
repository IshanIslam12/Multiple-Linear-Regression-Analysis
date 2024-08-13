

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


data = pd.read_csv('marketing_sales_data.csv')

data.head()




# **Question:** What are some purposes of EDA before constructing a multiple linear regression model?

# Potential reasons include:
# 
# * Understanding which variables are present in the data
# * Reviewing the distribution of features, such as minimum, mean, and maximum values
# * Plotting the relationship between the independent and dependent variables to visualize which features have a linear relationship
# * Identifying issues with the data, such as incorrect values (e.g., typos) or missing values


sns.pairplot(data)


# **Question:** Which variables have a linear relationship with `Sales`? Why are some variables in the data excluded from the preceding plot?

# `Radio` and `Social Media` both appear to have linear relationships with `Sales`. Given this, `Radio` and `Social Media` may be useful as independent variables in a multiple linear regression model estimating `Sales`. 
# `TV` and `Influencer` are excluded from the pairplot because they are not numeric. 


print(data.groupby('TV')['Sales'].mean())
print('')


print(data.groupby('Influencer')['Sales'].mean())


# **Question:** What do you notice about the categorical variables? Could they be useful predictors of `Sales`?

# The average `Sales` for `High` `TV` promotions is considerably higher than for `Medium` and `Low` `TV` promotions. `TV` may be a strong predictor of `Sales`.
# The categories for `Influencer` have different average `Sales`, but the variation is not substantial. `Influencer` may be a weak predictor of `Sales`.
# These results can be investigated further when fitting the multiple linear regression model. 

data.dropna(axis=0)


data = data.rename(columns={'Social Media': 'Social_Media'})


ols_formula = 'Sales ~ C(TV) + Radio'
OLS = ols(formula = ols_formula, data = data)
model = OLS.fit()
model_results = model.summary()


model_results
# **Question:** Which independent variables did you choose for the model, and why?
# * `TV` was selected, as the preceding analysis showed a strong relationship between the `TV` promotional budget and the average `Sales`.
# * `Radio` was selected because the pairplot showed a strong linear relationship between `Radio` and `Sales`.
# * `Social Media` was not selected because it did not increase model performance and it was later determined to be correlated with another independent variable: `Radio`.
# * `Influencer` was not selected because it did not show a strong relationship to `Sales` in the preceding analysis.



fig, axes = plt.subplots(1, 2, figsize = (8,4))
sns.scatterplot(x = data['Radio'], y = data['Sales'],ax=axes[0])
axes[0].set_title("Radio and Sales")
sns.scatterplot(x = data['Social_Media'], y = data['Sales'],ax=axes[1])
axes[1].set_title("Social Media and Sales")
axes[1].set_xlabel("Social Media")


plt.tight_layout()

# **Question:** Is the linearity assumption met?
# The linearity assumption holds for `Radio`, as there is a clear linear relationship in the scatterplot between `Radio` and `Sales`. `Social Media` was not included in the preceding multiple linear regression model, but it does appear to have a linear relationship with `Sales`.


residuals = model.resid
fig, axes = plt.subplots(1, 2, figsize = (8,4))


sns.histplot(residuals, ax=axes[0])
axes[0].set_xlabel("Residual Value")
axes[0].set_title("Histogram of Residuals")


sm.qqplot(residuals, line='s',ax = axes[1])
axes[1].set_title("Normal QQ Plot")
plt.tight_layout()
plt.show()




fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
fig.set_title("Fitted Values v. Residuals")

fig.axhline(0)


plt.show()
# **Question:** Is the constant variance assumption met?
# The fitted values are in three groups because the categorical variable is dominating in this model, meaning that TV is the biggest factor that decides the sales. However, the variance where there are fitted values is similarly distributed, validating that the assumption is met

sns.pairplot(data)


from statsmodels.stats.outliers_influence import variance_inflation_factor
X = data[['Radio','Social_Media']]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])
df_vif

# **Question 8:** Is the no multicollinearity assumption met?

# The preceding model only has one continous independent variable, meaning there are no multicollinearity issues.  ​
# If a model used both `Radio` and `Social_Media` as predictors, there would be a moderate linear relationship between `Radio` and `Social_Media` that violates the multicollinearity assumption. Furthermore, the variance inflation factor when both `Radio` and `Social_Media` are included in the model is 5.17 for each variable, indicating high multicollinearity.



model_results


# **Question:** What is your interpretation of the model's R-squared?
# Using `TV` and `Radio` as the independent variables results in a multiple linear regression model with $R^{2} = 0.904$. In other words, the model explains $90.4\%$ of the variation in `Sales`. This makes the model an excellent predictor of `Sales`. 

# **Question:** What are the model coefficients?
# When `TV` and `Radio` are used to predict `Sales`, the model coefficients are:
# * $\beta_{0} =  218.5261$
# * $\beta_{TVLow}= -154.2971$
# * $\beta_{TVMedium} = -75.3120$
# * $\beta_{Radio} =  2.9669$

# **Question:** How would you write the relationship between `Sales` and the independent variables as a linear equation?
# $\text{Sales} = \beta_{0} + \beta_{1}*X_{1}+ \beta_{2}*X_{2}+ \beta_{3}*X_{3}$
# $\text{Sales} = \beta_{0} + \beta_{TVLow}*X_{TVLow}+ \beta_{TVMedium}*X_{TVMedium}+ \beta_{Radio}*X_{Radio}$
# $\text{Sales} = 218.5261 - 154.2971*X_{TVLow} - 75.3120*X_{TVMedium}+ 2.9669 *X_{Radio}$

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?
# The default `TV` category for the model is `High` since there are coefficients for the other two `TV` categories, `Medium` and `Low`. Because the coefficients for the `Medium` and `Low` `TV` categories are negative, that means the average of sales is lower for `Medium` or `Low` `TV` categories compared to the `High` `TV` category when `Radio` is at the same level.
# For example, the model predicts that a `Low` `TV` promotion is 154.2971 lower on average compared to a `high` `TV` promotion given the same `Radio` promotion.
# The coefficient for `Radio` is positive, confirming the positive linear relationship shown earlier during the exploratory data analysis.
# The p-value for all coefficients is $0.000$, meaning all coefficients are statistically significant at $p=0.05$. The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. 
# For example, there is a $95\%$ chance that the interval $[-163.979,-144.616]$ contains the true parameter of the slope of $\beta_{TVLow}$, which is the estimated difference in promotion sales when a `Low` `TV` promotion is chosen instead of a `High` `TV` promotion.[Write your response here. Double-click (or enter) to edit.]

# **Question:** Why is it important to interpret the beta coefficients?
# Beta coefficients allow you to estimate the magnitude and direction (positive or negative) of the effect of each independent variable on the dependent variable. The coefficient estimates can be converted to explainable insights, such as the connection between an increase in TV promotional budgets and sales mentioned previously.

# **Question:** What are you interested in exploring based on your model?
# * Providing the business with the estimated sales given different TV promotions and radio budgets
# * Additional plots to help convey the results, such as using the `seaborn` `regplot()` to plot the data with a best fit regression line

# **Question:** Do you think your model could be improved? Why or why not? How?
# Yes,  by getting a more granular view of the `TV` promotions, such as by considering more categories or the actual `TV` promotional budgets, and getting additional variables, such as the location of the marketing campaign or the time of year, could increase model accuracy. 



# **What are some key takeaways that you learned from this lab?** 
# * Multiple linear regression is a powerful tool to estimate a dependent continous variable from several independent variables.
# * Exploratory data analysis is useful for selecting both numeric and categorical features for multiple linear regression.
# * Fitting multiple linear regression models may require trial and error to select variables that fit an accurate model while maintaining model assumptions.
 
# **What findings would you share with others?** 
# According to the model, high TV promotional budgets result in significantly more sales than medium and low TV promotional budgets. For example, the model predicts that a `Low` `TV` promotion is 154.2971 lower on average than a `high` `TV` promotion given the same `Radio` promotion. 
# The coefficient for radio is positive, confirming the positive linear relationship shown earlier during the exploratory data analysis.
# The p-value for all coefficients is $0.000$, meaning all coefficients are statistically significant at $p=0.05$. The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. 
# For example, there is a $95\%$ chance the interval $[-163.979,-144.616]$ contains the true parameter of the slope of $\beta_{TVLow}$, which is the estimated difference in promotion sales when a low TV promotional budget is chosen instead of a high TV promotion budget.
# **How would you frame your findings to stakeholders?**
# High TV promotional budgets have a substantial positive influence on sales. The model estimates that switching from a high to medium TV promotional budget reduces sales by $\$75.3120$ million (95% CI $[-82.431,-68.193])$, and switching from a high to low TV promotional budget reduces sales by $\$154.297$ million (95% CI $[-163.979,-144.616])$. The model also estimates that an increase of $\$1$ million in the radio promotional budget will yield a $\$2.9669$ million increase in sales (95% CI $[2.551,3.383]$).
# Thus, it is recommended that the business allot a high promotional budget to TV when possible and invest in radio promotions to increase sales. 

