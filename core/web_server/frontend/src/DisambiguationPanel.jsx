import React, { useContext } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AstroFlowContext from './AstroFlowContext.jsx';
import StackCard from "./StackCard.jsx";
import TargetCard from "./TargetCard.jsx";
import MosaicCard from "./MosaicCard.jsx";
import CalibrationStackCard from "./CalibrationStackCard.jsx";

export default function DisambiguationPanel({ items_in }) {
	const { stacks, targets, mosaics, calibrationStacks, isForeign, showForeign } = useContext(AstroFlowContext);

	const lookup = (item) => {
		const [type, id] = item.split(",");
		const collection = {
			stack: stacks, target: targets, mosaic: mosaics,
			calibration_stack: calibrationStacks,
		}[type];
		return collection?.[id];
	};

	// Most recent first: search results and long card lists are usually
	// hunted for the newest data (e.g. last night's flats).
	const newestDate = (item) => {
		const t = Date.parse(lookup(item)?.newest_date);
		return Number.isNaN(t) ? 0 : t;
	};

	// const params = useParams();
	let { items } = useParams();
	items = items || items_in;
	if (!items || items.trim() === "") {
		return (
			<div style={{
				margin: "2em",
				padding: "2em",
				textAlign: "center",
				color: "#999",
				fontStyle: "italic",
				fontSize: "1.25em",
			}}>
				No results found.
			</div>
		);
	}

	// A disambiguation URL is a snapshot: objects in it can vanish as
	// identification evolves (deleted frames, re-keyed stacks). Render
	// what still exists and say how many links went stale, instead of
	// letting a card crash on a missing object.
	const allItems = items.split(";");
	const existingItems = allItems.filter(item => lookup(item) !== undefined);
	const missingCount = allItems.length - existingItems.length;
	const presentItems = showForeign
		? existingItems
		: existingItems.filter(item => !isForeign(...item.split(",")));
	const hiddenForeignCount = existingItems.length - presentItems.length;

	return (
		<>
			{missingCount > 0 && (
				<div style={{ margin: "1em", color: "#999", fontStyle: "italic" }}>
					{missingCount} result{missingCount === 1 ? "" : "s"} from this
					search link no longer exist{missingCount === 1 ? "s" : ""}.
				</div>
			)}
			{hiddenForeignCount > 0 && (
				<div style={{ margin: "1em", color: "#999", fontStyle: "italic" }}>
					{hiddenForeignCount} foreign result{hiddenForeignCount === 1 ? "" : "s"} hidden.
				</div>
			)}
			<div style={{
				marginLeft: "1em",
				marginRight: "1em",
				display: "grid",
				gridTemplateColumns: "repeat(4, 1fr)",
				gridColumnGap: "1em",
				gridRowGap: "1em",
			}}>
				{presentItems.sort((a, b) => newestDate(b) - newestDate(a)).map(item => {
					let [type, id] = item.split(",");
					if (type === "stack") {
						return <StackCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							stack_id={id}
						/>;
					} else if (type === "target") {
						return <TargetCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							target_id={id}
						/>;
					} else if (type === "mosaic") {
						return <MosaicCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							mosaic_id={id}
						/>;
					} else if (type === "calibration_stack") {
						return <CalibrationStackCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							stack_id={id}
						/>;
					}
				}
				)}
			</div>
		</>
	);
}


