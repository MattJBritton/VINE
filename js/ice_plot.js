(function(global, d3, _) {

  const DEFAULT_CURVE = d3.curveLinear;
  const DEFAULT_THUMBNAIL = true;
  const DEFAULT_MARGIN_PROPS = {
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  };
  const DEFAULT_AXIS_PROPS = {
    top: false,
    right: false,
    bottom: true,
    left: true,
  };

  const DEFAULT_HISTOGRAM_PROPS = {
    outer_width: 600,
    outer_height: 600,
    top: 90,
    right: 70,
    bottom: 50,
    left: 120,
  }


  function seriesExtent(series, accessor) {
    let gMin;
    let gMax;
    let nMin;
    let nMax;
    let n = series.length;
    let i = -1;
    while (++i < n) {
      // Get the initial extent.
      ([gMin, gMax] = d3.extent(series[i], accessor));
      while (++i < n) {
        // Compare remaining extents.
        ([nMin, nMax] = d3.extent(series[i], accessor));
        if (nMin < gMin) { gMin = nMin; }
        if (nMax > gMax) { gMax = nMax; }
      }
    }
    return [gMin, gMax];
  }


  /**
   * Create a new ice_plot generator.
   */
  global.ice_plot = function() {

    let marginProps = Object.assign({}, DEFAULT_MARGIN_PROPS);
    let axisProps = Object.assign({}, DEFAULT_AXIS_PROPS);
    let histogramProps = Object.assign({}, DEFAULT_HISTOGRAM_PROPS);
    let curve = DEFAULT_CURVE;
    let bolThumbnail = DEFAULT_THUMBNAIL;
    const localId = d3.local();
    const localScales = d3.local();
    const localSVG = d3.local();
    const localSidebar = d3.local();

    const dispatch = d3.dispatch(
      'maximize',
      'minimize'
    );    

    function ice_plot(svgSelection) {

      svgSelection.each(function(packed_data) {

        //unpack data
        data = packed_data[0];
        distribution_data = packed_data[1];

        // Set the chart ID.
        if (localId.get(this) === undefined) { 
          localId.set(this, _.uniqueId('ice_plot')); }

        // Calculate chart properties.
        const svg = d3.select("#svg_"+(bolThumbnail?getSafeFeatureName(data["feature_name"]):"main"));
        const sidebar = d3.select("#div_"+(bolThumbnail? getSafeFeatureName(data["feature_name"]):"main")+"_sidebar")
        const props = getProps(svg);
        const scales = getScales(data, props);
        const axes = getAxes(scales);

        scaledHistogramProps = {}
        for(var key in histogramProps) {
          scaledHistogramProps[key]= histogramProps[key]/data["clusters"].length;
        }        
        scaledHistogramProps["width"] = scaledHistogramProps["outer_width"]
        -(scaledHistogramProps["left"]+scaledHistogramProps["right"]);
        scaledHistogramProps["height"] = scaledHistogramProps["outer_height"]
        -(scaledHistogramProps["top"]+scaledHistogramProps["bottom"]);

        // Persist the scales locally
        localScales.set(this, scales);
        localSVG.set(this, svg);
        localSidebar.set(this, sidebar);

        // Render the chart skeleton.
        renderChart(localSVG.get(this), props, data);
        renderAxes(localSVG.get(this), props, axes);
        renderAxisLabels(localSVG.get(this), props, data);

        // Render the chart content.
        renderPDP(localSVG.get(this), props, scales, data);
        renderClusters(localSVG.get(this), props, scales, data);

        // Render the sidebar
        renderSidebar(localSidebar.get(this), data, 
          distribution_data, scaledHistogramProps);
      });
    }

    function getProps(svg) {
      const svgEl = svg.node();
      const width = svgEl.clientWidth;
      const height = svgEl.clientHeight;
      const chartWidth = width - marginProps.left - marginProps.right;
      const chartHeight = height - marginProps.top - marginProps.bottom;

      return {
        width,
        height,
        chartWidth,
        chartHeight,
        margin: marginProps,
      };
    }

    function getScales(data, props) {

      const xDomain = d3.extent(data["x_values"]);
      curves_for_y_domain = data["clusters"].map(d => d.line);
      curves_for_y_domain.push(data["pdp_line"]);
      const yDomain = seriesExtent(curves_for_y_domain, 
        d=>d);  
      const lineWidthDomain = [0,d3.max(data["clusters"].map(
        d => d.cluster_size))];             
      const xRange = [0, props.chartWidth];
      const yRange = [props.chartHeight, 0];
      const lineWidthRange = [1,5];

      return {
        x: d3.scaleLinear().domain(xDomain).range(xRange),
        y: d3.scaleLinear().domain(yDomain).range(yRange),
        lineWidth: d3.scaleLinear().domain(lineWidthDomain)
          .range(lineWidthRange)
      };
    }

    function getAxes(scales) {
      const axes = [];

      if (axisProps.top) {
        axes.push({ cls:'top', axis: d3.axisTop(scales.x) });
      }
      if (axisProps.right) {
        axes.push({ cls:'right', axis: d3.axisRight(scales.y) });
      }
      if (axisProps.bottom) {
        axes.push({ cls:'bottom', axis: d3.axisBottom(scales.x) });
      }
      if (axisProps.left) {
        axes.push({ cls:'left', axis: d3.axisLeft(scales.y) });
      }
      return axes;
    }

    /**
     * Render the chart skeleton, binding data to the appropriate content areas.
     * @param {object} svg The SVG selection
     * @param {object} props The chart properties
     * @param {object} data The data model
     */
    function renderChart(svg, props, data) {
      // Render the clipping path.
      const clipUrl = renderClipPath(svg, props);

      // Render the series area. Clip.
      const seriesContainer = renderContainer(svg, props, 'series-content');

      // Render the axis container. Do not clip.
      const axisContainer = renderContainer(svg, props, 'axis-content');

      // Set up onclick functionality to open a thummbnail
      if(bolThumbnail) {
        svg.on("click", _.partial(maximize, svg, data));
      } else {
        closeButtonGrp = svg.append("g")
          .classed("closeButton", true)
          .attr('transform', d =>
            `translate(${props.width-40}, ${30})`    
          );
          closeButtonGrp.append("button")
            .attr("class","btn btn-primary")
            .attr("height", 25)
            .attr("width", 20)
            .attr("cursor", "pointer")
            .on("click", _.partial(minimize, svg));
        // closeButtonGrp.append("rect")
        //   .attr("height", 25)
        //   .attr("width", 20)
        //   .attr("fill", "red")
        //   .attr("opacity", .3)
        //   .attr("cursor", "pointer")
        //   .on("click", _.partial(minimize, svg));

        // closeButtonGrp.append("text")
        //   .attr("y", 20)
        //   .attr("x", 5)
        //   .attr("font-size", 20)
        //   .text("X")
        //   .attr("cursor", "pointer")
        //   .on("click", _.partial(minimize, svg));
      }
    }

    /**
     * Render the clipping path for the SVG.
     * @param {object} svg The SVG selection
     * @param {object} props The chart properties
     * @return {string} The URL for referencing the clip path
     */
    function renderClipPath(svg, props) {
      const id = localId.get(svg.node());
      let defs = svg
        .selectAll('defs')
        .data([0]);
      defs = defs
        .enter()
        .append('defs')
        .merge(defs);
      let clipPath = defs
        .selectAll('clipPath')
        .data([0])
      clipPath = clipPath
        .enter()
        .append('clipPath')
          .attr('id', `clip-${id}`)
        .merge(clipPath);
      let clipRect = clipPath
        .selectAll('rect')
        .data([0]);
      clipRect = clipRect
        .enter()
        .append('rect')
        .merge(clipRect)
          .attr('width', props.chartWidth)
          .attr('height', props.chartHeight);
      return `url(#clip-${id})`;
    }

    /**
     * Render a container for SVG content.
     * @param {object} svg The SVG selection
     * @param {object} props The chart properties
     * @param {string} cls The container class/label
     * @param {string} clipUrl The clip path URL. If given, the container content will be clipped
     */
    function renderContainer(svg, props, cls, clipUrl) {
      const update = svg
        .selectAll(`.${cls}`)
        .data([0]);
      const enter = update
        .enter()
        .append('g')
          .attr('class', cls);
      if (clipUrl) {
        enter.attr('clip-path', clipUrl);
      }
      return enter
        .merge(update)
          .attr('transform', `translate(${props.margin.left}, ${props.margin.top})`);
    }
    
    /**
     * Render the chart axes.
     * @param {object} svg The SVG selection
     * @param {object} props The chart properties
     * @param {array<object>} axes The chart axes properties
     */
    function renderAxes(svg, props, axes) {
      const container = svg.select('.axis-content');
      const update = container
        .selectAll('.axis')
        .data(axes);
      const enter = update
        .enter()
        .append('g')
          .attr('class', d => `axis ${d.cls}`)
          .attr('transform', d => {
            if (d.cls === 'right') {
              return `translate(${props.chartWidth}, 0)`;
            } else if (d.cls === 'bottom') {
              return `translate(0, ${props.chartHeight})`;
            }
            return `translate(0,0)`;
          });
      const exit = update
        .exit()
        .remove();
      enter
        .merge(update)
        .each(function(d) { d3.select(this).call(d.axis); });
    }

    // from https://stackoverflow.com/questions/11189284/d3-axis-labeling#11194968
    function renderAxisLabels(svg, props, data) {

      svg.append("text")
        .attr("class", "x label")
        .attr("text-anchor", "middle")
        .attr("x", props.width/2)
        .attr("y", props.height -10)
        .text(data["feature_name"]);

      svg.append("text")
        .attr("class", "y label")
        .attr("text-anchor", "start")
        .attr("x", 5)
        .attr("y", props.height/2)
        .attr("dy", ".75em")
        //.attr("transform", "rotate(-90)")
        .text("𝚫ŷ");
    }

    /**
     * Render the PDP line
     * @param {object} svg The SVG selection
     * @param {object} props The chart properties
     * @param {object} scales The chart scales
     * @param {object} data The data model
     */
    function renderPDP(svg, props, scales, data) {

      const {
        x: xScale,
        y: yScale
      } = scales;

      let line = d3.line()
        .x((d, i) => xScale(data["x_values"][i]))
        .y((d, i) => yScale(d))
        .curve(curve);
      const container = svg.select('.series-content');
      let series = container
        .selectAll('.pdp_curve')
        .data([data["pdp_line"]]);
      series.exit().remove();

      series = series
        .enter()
        .append('path')
          .attr('class', 'pdp_curve')
          .attr('fill', 'none')
          .attr('stroke', "black")
          .attr('stroke-width', 5)
          .attr('opacity', 1)
        .merge(series)
          .attr('d', d => line(d));
    }


    function renderClusters(svg, props, scales, data) {

      const {
        x: xScale,
        y: yScale,
        lineWidth: widthScale
      } = localScales.get(svg.node());

      const line = d3.local()
      line.set(svg.node(), d3.line()
        .x((d, i) => xScale(data["x_values"][i]))
        .y((d, i) => yScale(d))
        .curve(curve));
      const container = svg.select('.series-content');
      let cluster_gs = container
        .selectAll('.cluster')
        .data(data["clusters"]);

      cluster_gs = cluster_gs
        .enter()
        .append("g")
        //.merge(cluster_gs)
          .attr("class", "cluster")
          .attr("id", (d,i) => "cluster_"+i);          

      let cluster_curves = cluster_gs
        .append('path')
        .merge(container.selectAll(".cluster_curve"))
          .attr("class", "cluster_curve")
          .attr('fill', 'none')
          .attr('stroke', "blue")
          .attr("stroke-width", d => widthScale(d.cluster_size))
          .attr('opacity', 0.5)
          .attr('d', d => line.get(svg.node())(d.line));

      //to increase click target size
      let invisible_cluster_curves = cluster_gs
        .append('path')
        .merge(container.selectAll(".cluster_curve"))
          .attr("class", "cluster_curve")
          .attr('fill', 'none')
          .attr('stroke', "blue")
          .attr("stroke-width", 15)
          .attr("opacity", 0)
          .attr("cursor", "pointer")
          .attr('d', d => line.get(svg.node())(d.line))
          .attr("show_member_curves", false)
          .on("click", (d,i)=> {
              if(!bolThumbnail) {
                d3.event.stopPropagation();
                this.show_member_curves = !this.show_member_curves;
                show_member_curves(line, svg, container, d,i, 
                  this.show_member_curves);
              }
          });
    }

    function renderSidebar(div, data, distribution_data, scaledHistogramProps){     

      //adapted from http://bl.ocks.org/jfreels/6734025
      //columns = Object.keys(distribution_data);
      columns = [...new Set(data["clusters"].map(d=> d.split_feature))]; 
      num_clusters = data["clusters"].length;
      sorted_clusters = data["clusters"].sort((a,b) => {
        return b["line"][b["line"].length-1] - a["line"][a["line"].length-1];})
      var table = div.append("table")
        .attr("id", "cluster_definition_table")
        .style("border", "1px solid black")
        .style("border-spacing", "10px")
        .style("border-collapse", "collapse");
      var thead = table.append('thead');
      var tbody = table.append('tbody');  
      thead.append('tr')
        .selectAll('th')
        .data(columns).enter()
        .append('th')
          .style("border", "1px solid black")
          .text(function (column) { return column; }); 

      var rows = tbody.selectAll('tr')
        .data(sorted_clusters)
        .enter()
        .append('tr')
          .style("border", "1px solid black")
          .style("padding", "5px");

      // create a cell in each row for each column
      var cells = rows.selectAll('td')
        .data(function (row) {
          return columns.map(function (column) {
            return {
              feature: column, 
              column: distribution_data[column], 
              row: row};
          });
        })
        .enter()
        .append('td')
          .style("border", "1px solid black")
          .style("padding", "5px");

      cells
        .append("svg")
          .attr("class", "svg_hist")
          .attr("width", scaledHistogramProps.outer_width)
          .attr("height", scaledHistogramProps.outer_height)
        .each( function(d,i) {

          if(d["feature"] == d["row"]["split_feature"]) {
            //histogram adapted from http://bl.ocks.org/nnattawat/8916402
            var x_scale = d3.scaleLinear()
                  .domain(d3.extent(d["column"].map(e=> e.x)))
                  .range([0, scaledHistogramProps.width]); 
            var y_scale = d3.scaleLinear()
                  .domain(d3.extent(d["column"].map(e=> e.y)))
                  .range([scaledHistogramProps.height, 0]);               

            var bar = d3.select(this).selectAll(".bar")
                .data(d["column"])
              .enter().append("g")
                .attr("class", "bar")
                .attr("transform", e=> {
                  return "translate(" + (x_scale(e.x)) + ","
                  + (y_scale(e.y) + scaledHistogramProps.bottom + scaledHistogramProps.top ) + ")"; });

            bar.append("rect")
                .attr("y", -1*scaledHistogramProps.bottom)
                .attr("x", scaledHistogramProps.left)
                .attr("width", .5*scaledHistogramProps.width/(d["column"].length))
                .attr("height", e=> scaledHistogramProps.height - y_scale(e.y))
                .attr("fill", e=> {
                  if(d["row"]["split_direction"] == ">") {
                    return e.x>d["row"]["split_val"]?"steelblue":"blue";
                  } else {
                    return e.x<=d["row"]["split_val"]?"steelblue":"blue";
                  }
                });

            if(d["column"].length > 10) {
              x_axis = d3.axisBottom(x_scale).ticks(10);
            } else {
              x_axis = d3.axisBottom(x_scale).tickValues(
                    d["column"].map(e=> e.x));
            }

            d3.select(this).append("g")
              .attr("class", "x axis")
              .attr("transform", "translate(" + scaledHistogramProps.left + ","
                + (scaledHistogramProps.outer_height - scaledHistogramProps.bottom) + ")")
              .call(x_axis);  

            d3.select(this).append("g")
                .attr("class", "y axis")
                .attr("transform", "translate("
                  + scaledHistogramProps.left + "," + scaledHistogramProps.top + ")")
                .call(d3.axisLeft(y_scale));    

            d3.select(this).append('text')
              .attr("class", "cluster_label")
              .attr("text-anchor", "end")
              .attr("x", scaledHistogramProps.outer_width)
              .attr("y", 0)
              .attr("dy", ".75em")
              .attr("font-size", ".75em")
              .text(d["row"]["split_feature"] + d["row"]["split_direction"]
                + d["row"]["split_val"]);   

            d3.select(this).append('text')
              .attr("class", "cluster_label")
              .attr("text-anchor", "end")
              .attr("x", scaledHistogramProps.outer_width)
              .attr("y", scaledHistogramProps.top/2)
              .attr("dy", ".75em")
              .attr("font-size", ".75em")
              .text("Accuracy: " + d["row"]["accuracy"] + "%");    

            d3.select(this).append('text')
              .attr("class", "cluster_label")
              .attr("text-anchor", "end")
              .attr("x", scaledHistogramProps.outer_width)
              .attr("y", scaledHistogramProps.top)
              .attr("dy", ".75em")
              .attr("font-size", ".75em")
              .text("Cluster Size: " + d["row"]["cluster_size"]);                                        
            }                 
          }
        );
    }

    function show_member_curves(line, svg, container, d_cluster, i_cluster, bolShow) {

      if(bolShow) {

        container
          //.select("#cluster_"+i_cluster)
          .selectAll(".member_curve .cluster_"+i_cluster)
          .data(d_cluster["individual_ice_curves"])
          .enter()
          .append('path')
            .attr("class", "member_curve cluster_"+i_cluster)
            //.attr("class", "cluster_"+i_cluster)
            .attr("id", (d,i)=>"member_curve_"+i_cluster+"."+i)
            .attr('fill', 'none')
            .attr('stroke', "red")
            .attr("stroke-width", .7)
            .attr('opacity', 0.3)
            .attr('d', d => line.get(svg.node())(d));
      } else {

        container.selectAll(".member_curve").filter(".cluster_"+i_cluster)
        .remove();        
      }
    }

    function maximize(svg, data) {
      console.log("maximize");
      dispatch.call("maximize", svg.node(), data["feature_name"]);    
    }

    function minimize(svg) {
      console.log("minimize");
      dispatch.call("minimize", svg.node());  
    }

    function getSafeFeatureName(feature) {

      return _.replace(feature,new RegExp(" ","g"),"_");
    }


    ice_plot.margin = function(m) {
      if (m === null) {
        marginProps = Object.assign({}, DEFAULT_MARGIN_PROPS);
        return ice_plot;
      } else if (typeof m === 'number') {
        marginProps = {
          top: m,
          right: m,
          bottom: m,
          left: m,
        };
        return ice_plot;
      } else if (typeof m === 'object') {
        marginProps = Object.assign({}, DEFAULT_MARGIN_PROPS, m);
        return ice_plot;
      }
      return marginProps;
    }


    ice_plot.axis = function(a) {
      if (a === null) {
        axisProps = Object.assign({}, DEFAULT_AXIS_PROPS);
        return ice_plot;
      } else if (a !== undefined) {
        axisProps = Object.assign({}, DEFAULT_AXIS_PROPS, a);
        return ice_plot;
      }
      return axisProps;
    }

    ice_plot.histogram = function(a) {
      if (a === null) {
        histogramProps = Object.assign({}, DEFAULT_HISTOGRAM_PROPS);
        return ice_plot;
      } else if (a !== undefined) {
        histogramProps = Object.assign({}, DEFAULT_HISTOGRAM_PROPS, a);
        return ice_plot;
      }
      return histogramProps;
    }

    ice_plot.curve = function(c) {
      if (c === null) {
        curve = DEFAULT_CURVE;
        return ice_plot;
      } else if (c !== undefined) {
        curve = c;
        return ice_plot;
      }
      return curve;
    }

    ice_plot.thumbnail = function(a) {
      if (a === null) {
        bolThumbnail = DEFAULT_THUMBNAIL;
        return ice_plot;
      } else if (a !== undefined) {
        bolThumbnail = a;
        return ice_plot;
      }
      return bolThumbnail;
    }    

    /**
     * Add, remove, or get the callback for the specified event types. See d3-dispatch.on.
     */
    ice_plot.on = function() {
      const value = dispatch.on.apply(dispatch, arguments);
      return value === dispatch ? ice_plot : value;
    };    

    return ice_plot;
  }

})(window, window.d3, window._);