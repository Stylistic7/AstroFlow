import React from "react";
import { useNavigate } from "react-router-dom";
import "./Starmap.css";
import { ReactPhotoSphereViewer } from 'react-photo-sphere-viewer';
import { MarkersPlugin } from '@photo-sphere-viewer/markers-plugin';
import "@photo-sphere-viewer/markers-plugin/index.css";


export default function Starmap({
	constellation_polylines,
	stacks,
	targets,
	mosaics,
}) {
	const navigate = useNavigate();

	// const handleClick = (event) => {
	// 	console.log('x ' + event.clientX + ', y ' + event.clientY);

	// 	let elements = document.elementsFromPoint(event.clientX, event.clientY);
	// 	let matching_items = [];
	// 	elements.forEach((elt, i) => {
	// 		console.log('clicked object ', elt.getAttribute("data"));

	// 		const data = elt.getAttribute("data");
	// 		if (data) {
	// 			matching_items.push(data);
	// 		}

	// 	});

	// 	navigate(`/disambiguation/${matching_items.join(";")}`);
	// };

	const handleClick = (data: events.ClickEvent & { type: "click"; }) => {
		console.log(data);
	}

	const copyright = <span id="starmap_copyright">
		&copy; ESA/Gaia/DPAC; CC BY-SA 3.0 IGO. Acknowledgement: A.
		Moitinho.
	</span>

	const plugins = [
		[MarkersPlugin, {
			markers: [],
		}],
	]

	const handleReady = (instance) => {
		fetch('/constellation_polylines')
			.then(r => r.json())
			.then(constellation_polylines => {
				const markersPlugs = instance.getPlugin(MarkersPlugin);
				if (!markersPlugs)
					return;

				constellation_polylines.map((polyline, index) => {
					markersPlugs.addMarker({
						id: "polyline_" + index,
						polyline: polyline,
						style: {
							stroke: 'white',
							strokeOpacity: '0.5',
							strokeWidth: '1px',
						},
					});
				});
			});

		fetch('/stacks')
			.then(r => r.json())
			.then(stacks => {
				const markersPlugs = instance.getPlugin(MarkersPlugin);
				if (!markersPlugs)
					return;

				for (let [id, stack] of Object.entries(stacks)) {
					if (!stack.projected_footprints) continue;

					markersPlugs.addMarker({
						id: "stack_" + id,
						polygon: stack.projected_footprints['galactic']['coords'],
						style: {
							stroke: 'rgba(0, 0, 200, 0.5)',
							strokeOpacity: '0.5',
							strokeWidth: '2px',
						},
					});
				};
			});

		fetch('/mosaics')
			.then(r => r.json())
			.then(mosaics => {
				const markersPlugs = instance.getPlugin(MarkersPlugin);
				if (!markersPlugs)
					return;

				for (let [id, mosaic] of Object.entries(mosaics)) {
					if (!mosaic.projected_footprint) continue;

					markersPlugs.addMarker({
						id: "mosaic_" + id,
						polygon: mosaic.projected_footprint['galactic']['coords'],
						style: {
							stroke: 'rgba(200, 0, 0, 0.8)',
							strokeOpacity: '0.8',
							strokeWidth: '4px',
						},
					});
				};
			});
	};

	return <ReactPhotoSphereViewer
		src="/static/images/ESA_Gaia_DR2_AllSky_Brightness_Colour_Cartesian_4000x2000.png"
		height={'100vh'}
		width={"100%"}
		plugins={plugins}
		maxFov={140}
		onReady={handleReady}
		onClick={handleClick}>
	</ReactPhotoSphereViewer>;
}
