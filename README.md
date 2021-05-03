# Disaster Response Pipeline

### Table of Contents

1. [Installation](#installation)
2. [Instruction](#instruction)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

All the libraries required for running the app are listed in the requirement.txt file.
For installing these libraries, only run this comand with Python 3 in root path:

```
$ pip install -r requirements.txt
```


## Instruction <a name="instruction"></a>

Follow all the instructions in this order to run the app.

1. Run in the project's root directory to run the ETL pipeline that cleans data and stores in database
	```
	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
	```

2. Run in the project's root directory to run the ML pipeline that trains classifier and saves
	```
	python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
	```

3. Run in the app's directory to run the Flask app:
	```
	$ python run.py
	```


## Project Motivation<a name="motivation"></a>

This a the second project for the Udacity Data Science Nanodegree programm. For this project I was interestested in building a machine learning pipeline to categorize emergency messages based on the needs communicate by the senders.


## File Descriptions <a name="files"></a>
<pre>
<code>.
├── <b>README.md</b>
├── <b>app</b> : Flask App Files
│ ├── <b>run.py</b> : Flask file to  run the app
│ └── <b>templates</b>
│ ├── <b>go.html</b>
│ └── <b>master.html</b>
├── <b>data</b> : It contains all ETL Files 
│ ├── <b>DisasterResponse.db</b> :  SQLite DataBase file containing cleaned data after ETL process  
│ ├── <b>disaster_categories.csv</b> :  Disaster Categories CSV file
│ ├── <b>disaster_messages.csv</b> : Messages CSV file
│ └── <b>process_data.py</b> : 
├── <b>models</b> : It contains all ML files
│ ├── <b>classifier.pkl</b> : classifier produced by train_classifier file
│ └── <b>train_classifier.py</b> : ML pipeline classification code
└── <b>requirements.txt</b>
 </code>
</pre>


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The Datasets used in this project is provided by figure-8 that contains real labeled disaster messages received by an aid organisation during disaster events.
