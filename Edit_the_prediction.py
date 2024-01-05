
class EditPredict:
    def __init__(self, predict):
        self.predict = predict

    def Edit(self):
        final = []
        for i in range(len(self.predict)):
            maximum = max(self.predict[i])
            if maximum == self.predict[i][0]:
                final.append(-1)
            elif maximum == self.predict[i][1]:
                final.append(0)
            elif maximum == self.predict[i][2]:
                final.append(1)
        print(final)
        return final


