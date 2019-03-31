(function(global, ice_plot, d3, _) {

  thumbProps = {
    chartHeight : 160,
    chartWidth : 185,
    x_margin : 20,
    y_margin : 20,
    svgs_per_row : 3
  }

  mainProps = {
    height : 900,
    width : 1100,
    chartHeight: 600,
    chartWidth: 700,
    sidebar_width : 75,
    x_margin : 30,
    y_margin : 30
  }

  loaded_data = [];
  nodes = [];
  links = [];
  featureXScale = d3.scaleLinear();
  featureYScale = d3.scaleLinear();
  forceLayoutComplete = false;

  function drawNode(n) {

    n.attr('r', 5)
      .attr('cx', function(d) { return d.x; })
      .attr('cy', function(d) { return d.y; });
  }

  function drawLink(l) {
    l.attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });
  }

  function updateForceLayoutPositions() {

    nodes.forEach(drawNode);

    links.forEach(drawLink);
  }

  function getSafeFeatureName(feature) {

    return _.replace(feature,new RegExp(" ","g"),"_");
  }

  function showFeature(feature, thumbnail, feature_num) {

    if(thumbnail) {
      props = thumbProps;
      name = feature;
      svg_css = {
        top: Math.floor(feature_num/props.svgs_per_row)
          * (props.height+props.y_margin), 
        left: (feature_num%props.svgs_per_row)
          * (props.width+props.x_margin), 
        position:'absolute'};  
      sidebar_css = {}
    } else {
      props = mainProps;
      name = 'main';
      svg_css = {
        top: 50, 
        left: 0, 
        position:'absolute'
      };
      sidebar_css = {
        top: 50, 
        left: mainProps.chartWidth, 
        position:'absolute'
      };     
    }

    //add new one
    var div = d3.select("#vis")
      .append("div")
        .attr("id", "div_"+getSafeFeatureName(name))
        .attr("height", props.chartHeight)
        .attr("width", props.chartWidth + thumbnail?0:props.sidebar_width);

    var svg = div
      .append("svg")
        .attr("id", "svg_"+getSafeFeatureName(name))
        .attr("class", "svg-feature")
        .attr("height", props.chartHeight)
        .attr("width", props.chartWidth);

    svg.append("rect")
      .attr("class","svg-border")
      .attr("x", 0)
      .attr("y", 0)
      .attr("rx", 20)
      .attr("ry", 20)
      .attr("height", props.chartHeight)
      .attr("width", props.chartWidth)

    if(!thumbnail) {
      var sidebar = div
        .append("div")
          .attr("id", "div_"+getSafeFeatureName(name)+"_sidebar")
          .attr("height", props.chartHeight)
          .attr("width", props.sidebar_width);
    }

    if(!thumbnail) {
      $("#svg_"+getSafeFeatureName(name)).css(svg_css);
      $("#div_"+getSafeFeatureName(name)+"_sidebar").css(sidebar_css);
    } else {
      d3.select("#vis").select("#svg_"+getSafeFeatureName(name))
        .style("position", "absolute")
        .style("left",
          featureXScale(loaded_data["features"][feature]["cluster_deviation"]))
        .style("top",
          featureYScale(loaded_data["features"][feature]["importance"]));     
    }
    // Create the ICE plot component.
    var plot = ice_plot()
      .margin({ top: 30, right: 30, left: 40, bottom: 50 })
      .axis({ bottom: true, left: true })
      .curve(d3.curveMonotoneX)
      .thumbnail(thumbnail)
      .on('maximize', feature => build_main(feature))
      .on('minimize', () => build_multiples());

    // Render the main timeline component.
    div
      .datum([loaded_data["features"][feature],
        loaded_data["distributions"]])
      .call(plot);     
  }

  function build_main(feature) {

    //drop existing charts
    d3.select("#vis").selectAll("div").remove();
    d3.select("#svg_Force").remove();
    showFeature(feature, false, null);
  }

  function build_multiples(){

    //drop existing charts
    d3.select("#vis").selectAll("div").remove();
    feature_num = 0;
    Object.keys(loaded_data["features"]).forEach(function(d){

      showFeature(d,true,feature_num);
      feature_num++;
    })
    build_force_layout();
    build_feature_axes();    
  }

  function build_feature_axes(){

    //X axis
    var featureXAxis = d3.axisTop(featureXScale)
      .tickValues(featureXScale.domain())
      .tickFormat((d,i) => {if(i==0){
        return "low";
      } else {
        return "high";
      }});

    var xG = d3.select("#vis").select("#svg_Force")
        //.attr("id", "svg_Axis")
      .append("g")
        .attr("width", 800)
        .attr("class", "x axis")
        .attr("transform", "translate(0,25)");

    xG
        .call(featureXAxis);         

    d3.select("#vis").select("#svg_Force").append("text")
      .attr("x", mainProps.width/2)
      .attr("y", 40)
      .attr("text-anchor", "middle")
      .text("Strength of Interaction Effects");

    //Y Axis
    var featureYAxis = d3.axisLeft(featureYScale)
      .tickValues([featureYScale.domain()[1], 
        (featureYScale.domain()[0] + featureYScale.domain()[1])/2,
        featureYScale.domain()[0]])
      .tickFormat((d,i) => {
        switch(i){
          case 0: 
            return "high";
            break;
          case 1:
            return "moderate";
            break;
          case 2:
            return "low";
            break;
        }
      });

    var yG = d3.select("#vis").select("#svg_Force")
        //.attr("id", "svg_Axis")
      .append("g")
        .attr("width", 800)
        .attr("class", "y axis")
        .attr("transform", "translate(50,0)");

    yG
        .call(featureYAxis);         

    d3.select("#vis").select("#svg_Force").append("text")
      .attr("class", "y label")
      .attr("text-anchor", "middle")
      .attr("y", 10)
      .attr("x", -mainProps.height/2)
      .attr("dy", ".75em")
      .attr("transform", "rotate(-90)")
      .text("Feature Impact");   
  }

  function build_force_layout(){

    if(forceLayoutComplete) {
      return;
    }
    
    //adapted from http://bl.ocks.org/sathomas/11550728
    //force layout
    var svg = d3.select("#vis")
      .append("svg")
        .attr("id", "svg_Force")
        .attr("height", mainProps.height)
        .attr("width", mainProps.width);

    var simulation = d3.forceSimulation()
    .alphaDecay(0.01)
    /*.force("link", 
      d3.forceLink()
        .id(d => d.name)
        .distance(d => 100*Math.log(d.value)))*/
    .force("charge", d3.forceManyBody()
        .strength(500)
        .distanceMin(2000)
        .distanceMax(1000000)
    )
    .force("center", d3.forceCenter(mainProps.width / 2, 
                                    mainProps.height / 2))
    .force('collision', d3.forceCollide()
        .radius(100));

    /*
    var link = svg.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter().append("line")
        .style("stroke", "black");*/

    var node = svg.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
        .attr("id", d=> d.name)
        .attr("r", 0)
        .attr("x", d=> d.x)
        .attr("y",d => d.y)
        .style("fill", "")
        .style("stroke-width", 40);

    /*var node = d3.select("#vis")
      .selectAll(".svg-feature")
      .data(nodes); */

    simulation
      .nodes(nodes)
      .on('tick', d=>{return tickedSVG(true);})
      .on('end', d=>{
        console.log("force layout complete");
        //forceLayoutComplete = true;
        return tickedSVG(true);
      });       

    /*simulation
    .force("link")
    .links(links);*/

    function tickedSVG(bolShow) {

      node._groups[0].forEach(d=>{
        
        d3.select("#vis").select("#svg_" + getSafeFeatureName(d.getAttribute("id")))
            .style("left", d.getAttribute("cx"))
            .style("top", d.getAttribute("cy"))
            .attr("visibility", bolShow?"visible":"hidden");        
      });

      node
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
    }     
  }  

  d3.json("static/data.json")  
  .then(data => {

    loaded_data = data;

    featureXScale = d3.scaleLinear()
    .domain(d3.extent(Object.values(loaded_data["features"]), d=> d["cluster_deviation"]))
    .range([mainProps.x_margin, mainProps.width-mainProps.x_margin]);

    featureYScale = d3.scaleLinear()
    .domain(d3.extent(Object.values(loaded_data["features"]), d=> d["importance"]))
    .range([mainProps.height-mainProps.y_margin, mainProps.y_margin]);

    //links = loaded_data["feature_distance_links"];
    nodes = Object.entries(loaded_data["features"])
      .map(d=> {return {"name":d[0],
                        "x": featureXScale(d[1]["cluster_deviation"]),
                        "y": featureYScale(d[1]["importance"])
                        }});

    build_multiples();
    });

}) (window, window.ice_plot, window.d3, window._);