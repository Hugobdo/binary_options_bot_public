![result](imgs/mac_forex.jpg)

This Project is under development. Any questions or suggestion can be sent to the Issues tab or by email.

## Contents
- [Contents](#contents)
- [Motivation](#motivation)
- [Setup](#setup)
- [Data](#data)
- [Getting\_started](#getting_started)
- [License](#license)

## Motivation

The purpose of this Artificial Intelligence project is to predict the result of the chosen binary option

This project is being developed as a hobby and for learning purposes

## Setup
Just open the directory cloned path in terminal and run
```shell
$ pip install -r requirements.txt
```
You can also use Pycharm and setup a virtual environment (Recommended).
You may need to directly use pip to install IQOption API from github.com/Lu-Yi-Hsun/iqoptionapi.git

## Data

The data comes from IQOption API, with the functions defined on api_wrapper.py, on the class ApiData

## Getting_started

To use the code maintained in this repository, put your iqoptions credentials on config.py file.
You will also have to either configure your mongoDB locally or use a cloud service, such as MongoDB Atlas.
If using the cloud service, you will have to put your connection string on config.py file as well.

After setting up your db, go to mongo_connector.py and run the "insert historical" function, to insert the historical data on your db.
With the data inserted, you can now run the robot.py script, to start the robot. It will use the best regressor model that was trained and saved on the models folder.
The robot will run on practice mode, unless you change it. You can also tune the robot parameters to try to get better results.
Keep in mind that everything is set to run on "EURUSD" binary option, 1 minute candles, 5 minutes expiration time. If you want to change it, you will have to change everything: the data collection, pre processing, model training, etc.

If you want to train new models, there are two scripts: regressor_train.py and classifier_train.py. They use the pre_processing.py and classifier_pre_processing.py scripts to prepare the data for training. On my tests, I only found good results with the regressor models, specifically with LSTM. The classifier models usually have an accuracy close to 50%.

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
