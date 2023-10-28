import './Starmap.css';

export default function Starmap({stacks, targets, mosaics}) {
	var svg = document.getElementById("starmap_svg");

	for (const [key, target] of Object.entries(targets)) {
		console.log('target', target);
		console.log('target.footprint', target['footprint']);

		var polygon = document.createElementNS(
			"http://www.w3.org/2000/svg", "polygon");
		polygon.classList.add("starmap_polygon");
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

	return (
		<div id="starmap_container">
			<img id="starmap_img" src="/static/images/starmap_2020_4k.png" />
			<img id="constellation_figures_img" src="/static/images/constellation_figures_4k.png" />
			<svg id="starmap_svg" viewBox="0 0 360 180"></svg>
		</div>
	);
}