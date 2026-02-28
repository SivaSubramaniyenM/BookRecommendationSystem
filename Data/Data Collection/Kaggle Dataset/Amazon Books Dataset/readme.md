# Amazon Books Dataset

- Read kaggle documentation and create key.
- Go to accounts--> settings--> create key - a json file will be downloaded which will have the username and api key.
- Inside the json file the credentials will be displayed in the dictionary format as follows
   `{"username":"srivatsalaka","key":"4387c6aa13bd61518a0370a07b936df1"}`
- pip install kaggle on terminal to use kaggle library
- By running this code, 3 datasets will be downloaded. But we doing to use only one among these three (books_df.csv)
- Columns = title, author, main genre, subgenre, type, price, rating, no of people who rated, url.
 
# Code explanation

1. **Importing Libraries**
   ```python
   import kaggle
   import json
   import os
   ```
   - **kaggle**: This library allows to interact with the Kaggle API.
   - **json**: This library is used to parse JSON files.
   - **os**: This library provides a way to interact with the operating system, including setting environment variables.

2. **Loading Credentials**
   ```python
   with open('C:\\Users\\kaana\\.kaggle\\kaggle.json') as config_file:
       config = json.load(config_file)
   ```
   - This code opens the `kaggle.json` file, which contains your Kaggle API credentials, and loads its contents into the `config` variable.

3. **Setting Environment Variables**
   ```python
   os.environ['KAGGLE_USERNAME'] = config['username']
   os.environ['KAGGLE_KEY'] = config['key']
   ```
   - Kaggle API credentials are set as environment variables. This is done to securely pass sensitive information like usernames and API keys to the Kaggle API.

4. **Authenticating with Kaggle API**
   ```python
   kaggle.api.authenticate()
   ```
   - This line authenticates session with the Kaggle API using the credentials stored in the environment variables.

5. **Downloading the Dataset**
   ```python
   dataset_url = 'chhavidhankhar11/amazon-books-dataset'
   kaggle.api.dataset_download_files(dataset_url, path='.', unzip=True)
   ```
   - This code specifies the dataset URL and downloads the dataset to the current directory, unzipping it in the process.

6. **Confirmation Message**
   ```python
   print("Datasets Downloaded Successfully !!!")
   ```
   - This line prints a confirmation message once the dataset is downloaded.

> [!NOTE]
> An environment variable is a dynamic value that can affect the way running processes behave on a computer.
> They are used to store configuration data and sensitive information like API keys, which can be accessed by programs during their execution.

# Challenges:

- First error msg ` OSError: Could not find kaggle.json. Make sure it's located in C:\Users\kaana\.kaggle. Or use the environment method.` 
- Chceked setup instructions at https://github.com/Kaggle/kaggle-api/ and copied the downloaded config file and pasted in the above locationa and also learned about environment variable.
- Checked what is environment method and found there is a library os and we can use os.environ to config the file.




