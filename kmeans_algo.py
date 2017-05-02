from flask import Flask, render_template, request
import pandas
from sklearn.cluster import KMeans
from flask import Flask
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='static')

@app.route('/kmeans', methods=['GET', 'POST'])
def get_kmeans_cluster():
	clusters = request.args.get('clusters','')
	data = pandas.read_csv("game.csv")
	k_model = KMeans(n_clusters = int(clusters), random_state=1)
	columns = data._get_numeric_data()
	k_model.fit(columns)
	labels = k_model.labels_
	pca_2 = PCA(2)
	plot = pca_2.fit_transform(columns)
	plt.scatter(x=plot[:, 0], y=plot[:, 1], c=labels)
	plt.savefig('static/kmeans_plot.png')
	return render_template('display.html')

@app.route('/')
def hello_world():
  return render_template('index.html')


if __name__ == '__main__':
  app.run()
