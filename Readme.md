### Visualisation for yearly income prediction

The interactive part is hosted on [Heroku](https://income-class-prediction.herokuapp.com/)
The original Adults project with data analysis and model selection is in my [Adults pet project GitHub repository](https://github.com/nadiia95/Adults_pet_project)

This project is an interactive part, so you are very welcome to visit Heroku and try it on your own

A short description of what this app does under the hood:
* the Random Forest Classification model is used for training
* data from the user is collected and transformed into the format, required by the model
* model predicts the income class
* the graph with Shapley values (feature importance) is built based on the prediction

Here a new 'Country' file is used to let the user select among all countries in the world. Custom column 'developed' is added with `true/false` values based on Table A of this United Nations [country classification](https://www.un.org/en/development/desa/policy/wesp/wesp_current/2014wesp_country_classification.pdf) report. This column is needed to transform the data to feed to the model.

The following data is collected from the user to make a prediction:
* Personal info
    * Age
    * Sex
    * Country
    * Ethnic group
* Family info
    * Marital Status
    * Family Belonging
* Education and professional info
    * Education level
    * Amount of working hours per week
    * Workclass
    * Occupation
* Capital operations Info
    * Capital gain
    * Capital loss
