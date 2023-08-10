const draw_footprints = (target_footprints) => {
	console.log('draw_footprints');
	console.log('target_footprints', target_footprints);

	var svg = document.getElementById("starmap_svg");

	for (const [target, footprint] of Object.entries(target_footprints)) {
		console.log('target', target);
		console.log('footprint', footprint);

		var polygon = document.createElementNS(
			"http://www.w3.org/2000/svg", "polygon");
		svg.appendChild(polygon);

		for (point of footprint) {
			var svg_point = svg.createSVGPoint();
		  svg_point.x = 360 - (point[0] + 180) % 360;
		  svg_point.y = 90 - point[1];
		  console.log('point', point[0], point[1]);
		  polygon.points.appendItem(svg_point);
		}
	}
};

fetch('/target_footprints')
   .then(response => response.json())
   .then(draw_footprints);