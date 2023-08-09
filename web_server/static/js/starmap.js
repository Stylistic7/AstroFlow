const draw_footprints = (footprints) => {
	console.log('draw_footprints');

	var svg = document.getElementById("starmap_svg");

	for (footprint of footprints) {
		console.log('new polygon')
		var polygon = document.createElementNS(
			"http://www.w3.org/2000/svg", "polygon");
		svg.appendChild(polygon);

		// footprint = [[190, 80], [170, 80], [170, -80], [190, -80]];

		for (point of footprint) {
			var svg_point = svg.createSVGPoint();
		  svg_point.x = 360 - (point[0] + 180) % 360;
		  svg_point.y = 90 - point[1];
		  console.log('point', point[0], point[1]);
		  polygon.points.appendItem(svg_point);
		}
	}
};

fetch('/footprints')
   .then(response => response.json())
   .then(draw_footprints);