import pickle


class SentimentClassifier(object):
    def __init__(self):
        input = "model.pkl"
        with open(input, "rb") as f:
            model = pickle.load(f)
        self.model = model
        self.classes_dict = {
            0: "Негативный отзыв",
            1: "Позитивный отзыв",
            -1: "Ошибка предсказания",
        }

    def predict_text(self, text):
        try:
            lst = []
            lst.append(text)
            return self.model.predict(lst)
        except:
            print("Ошибка предсказания")
            return -1

    def predict_text(self, text):
        lst = []
        lst.append(text)
        return text if text is None else self.model.predict(lst)

    def predict_list(self, list_of_texts):
        try:
            return self.model.predict(list_of_texts)
        except:
            print("Ошибка предсказания")
            return -1

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        return (
            prediction if prediction is None else self.classes_dict[int(prediction[0])]
        )
