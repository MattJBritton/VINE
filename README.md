# VINE: Visualizing Statistical Interactions in Black Box Models

* For detailed background, see paper on [https://arxiv.org/abs/1904.00561](arxiv.org)
* Run `main.py` to generate the file that the visualization consumes
    * `main.py` takes five arguments:
        * `dataset_name`: currently, choose from "bike", "diabetes", or "boston" (required)
        * 'num_clusters': number of clusters to generate per feature (default 5)
        * 'num_grid_points': number of points on the X-axis at which to predict a value for each curve. Higher means more granular but slower. (default 20)
        * `cluster_method`: "fast" or "good". (default "good")
        * `prune_clusters`: boolean. True returns a sparse set of important clusters (default True)
    * Example: run with `python main.py bike 5 20 good True`
* File is output to `static/data.json`
* `cd vis` to navigate to vis folder
* Launch webserver using `python -m SimpleHTTPServer 8000` or any method you prefer
* Open browser to `http://localhost:8000/`
    * Tested with Chrome v70
* Requirements:
    * Python 2.7
    * Numpy
    * Pandas
    * Scikit-learn
    * Javascript
    * D3.Js
    * Lodash
