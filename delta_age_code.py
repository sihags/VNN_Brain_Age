# Code to evaluate Delta-Age and brain age in OASIS-3 dataset
# df: Pandas dataframe consisting of diagnosis, chronological age data, and age residuals (VNN age estimate - chronological age) for all individuals.
# 'dx1' column in df stores the clinical diagnosis ('AD+' for individuals with AD dementia diagnosis and 'Cognitively normal' for HC group)
# 'residual' column in df stores (VNN age estimate - chronological age) for all individuals.

import statsmodels.api as sm

x_train = sm.add_constant(df.loc[df['dx1'] == 'Cognitively normal', 'Age'])
# regress age residual against age for HC group
lr = sm.OLS(df_y.loc[df_y['dx1'] == 'Cognitively normal','residual'], x_train).fit()
# Evaluate brain age after applying age bias correction to VNN estimates (stored in yhat_all variable)
df_y['corr_brain_age'] = yhat_all.squeeze().detach().numpy() - lr.predict(sm.add_constant(df_y['Age']))
# Evaluate Delta-Age
df_y['corr_brain_residual'] = df_y['corr_brain_age'] - df_y['Age']
