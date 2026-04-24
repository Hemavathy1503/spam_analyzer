
# Spam Analyzer

Spam Analyzer is a machine learning web application that detects whether a message is spam or not using NLP techniques.

## Features
- Spam detection with confidence score
- TF-IDF vectorization
- Naive Bayes model
- Flask-based web interface

## Tech Stack
- Python
- Flask
- scikit-learn
- pandas
- numpy

## Project Structure
```

spam-analyzer/
├── app.py
├── model.py
├── requirements.txt
├── spam.csv
├── templates/
├── static/

```

## Setup

1. Clone the repository
```

git clone [https://github.com/your-username/spam-analyzer.git](https://github.com/your-username/spam-analyzer.git)
cd spam-analyzer

```

2. Create virtual environment
```

python -m venv venv
venv\Scripts\activate

```

3. Install dependencies
```

pip install -r requirements.txt

```

4. Train model
```

python model.py

```

5. Run app
```

python app.py

```

6. Open in browser
```

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```

## Dataset
- SMS Spam Dataset
- 5574 messages
```

---
