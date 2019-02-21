# ICESkate: Feature Interactions using Interactive ICE Plots

* Run `main.py` to generate the file that the visualization consumes
    * `main.py` takes three arguments:
        * `dataset_name`: either "abalone" or "bike" (required)
        * 'num_clusters': number of clusters to generate per feature (default 5)
        * 'num_grid_points': number of points on the X-axis at which to predict a value for each curve. Higher means more granular but slower. (default 20)
    * Example: run with `python main.py bike`
* File is output to `vis/static/data.json`
* `cd vis` to navigate to vis folder
* Launch webserver using `python -m SimpleHTTPServer 8000` or any method your prefer
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