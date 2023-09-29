const draw_footprints = (targets) => {
	console.log('draw_footprints');
	console.log('targets', targets);

	var svg = document.getElementById("starmap_svg");

	for (const target of targets) {
		console.log('target', target);
		console.log('target.footprint', target['footprint']);

		var polygon = document.createElementNS(
			"http://www.w3.org/2000/svg", "polygon");
		svg.appendChild(polygon);
		polygon.addEventListener('click', () => console.log(target, 'clicked'));

		for (const point of target.footprint) {
		  var svg_point = svg.createSVGPoint();
		  svg_point.x = 360 - (point[0] + 180) % 360;
		  svg_point.y = 90 - point[1];
		  // console.log('point', point[0], point[1]);
		  polygon.points.appendItem(svg_point);
		}
	}
};

fetch('/targets')
   .then(response => response.json())
   .then(draw_footprints);