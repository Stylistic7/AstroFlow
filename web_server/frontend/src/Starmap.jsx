import React, { useState } from "react";

import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

import './Starmap.css';

export default function Starmap({constellation_polylines, stacks, targets, mosaics}) {
  // TODO: This probably isn't the right way to do this in React.
  // TODO: Every polling interval just adds more polylines and polygons.
	let svg = document.getElementById("constellations_svg");
	if (svg != null && svg.children.length == 0) {
		for (const cp of constellation_polylines) {
			let polyline = document.createElementNS(
				"http://www.w3.org/2000/svg", "polyline");
			polyline.classList.add("constellation_polyline");
			svg.appendChild(polyline);

			for (const point of cp) {
			  let svg_point = svg.createSVGPoint();
			  svg_point.x = 180 + point[0];
			  svg_point.y = 90 - point[1];
			  polyline.points.appendItem(svg_point);
			}
		}
	}

  // TODO: This probably isn't the right way to do this in React.
	svg = document.getElementById("starmap_svg");
	if (svg != null) {
		svg.innerHTML = "";

		for (const [key, target] of Object.entries(targets)) {
			let polygon = document.createElementNS(
				"http://www.w3.org/2000/svg", "polygon");
			polygon.classList.add("starmap_polygon");
			svg.appendChild(polygon);
			polygon.addEventListener('click', () => console.log(target, 'clicked'));

			for (const point of target.footprint_galactic) {
			  let svg_point = svg.createSVGPoint();
			  svg_point.x = 180 + point[0];
			  svg_point.y = 90 - point[1];
			  polygon.points.appendItem(svg_point);
			}
		}
	}

	return (
		<TransformWrapper>
      <TransformComponent>
      	<div id="starmap_container">
					<img id="starmap_img" src="/static/images/GAIA_EDR3.png" />
					<span id="starmap_copyright">&copy; ESA/Gaia/DPAC; CC BY-SA 3.0 IGO. Acknowledgement: A. Moitinho.</span>
					<svg id="constellations_svg" viewBox="0 0 360 180"></svg>
					<svg id="starmap_svg" viewBox="0 0 360 180"></svg>
				</div>
			</TransformComponent>
    </TransformWrapper>
	);
}