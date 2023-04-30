import requests


def predict():
    inference_request = {
        "dataframe_records": [[6.7, 3.3, 5.7, 2.1]]
    }

    endpoint = "http://localhost:1234/invocations"

    response = requests.post(endpoint, json=inference_request)

    print(response.text)


if __name__ == '__main__':
    predict()
