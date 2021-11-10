import pickle

#Parameters
f_vec = 'vectorizer.pkl'
f_lr = 'svm.pkl'

vec = pickle.load(open(f_vec, 'rb'))
lr = pickle.load(open(f_lr, 'rb'))

comments = open('input.txt', encoding='utf-8').readlines()
X = vec.transform(comments)

Y = lr.predict(X)
files = [open('file'+str(i)+'.txt', 'w', encoding='utf') for i in range(4)]

for idx, l in enumerate(Y):
    if l > 0:
        print(comments[idx], file=files[l-1], end='')
print('Done...')
