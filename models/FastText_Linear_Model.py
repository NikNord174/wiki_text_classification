from models.Linear_Model import Linear_Classifier


class FastText_Linear_Classifier(Linear_Classifier):
    def __init__(self, num_labels, devided_factor) -> None:
        super().__init__(num_labels, devided_factor)
        self.vector_size: int = 300
