import React, { useContext } from "react";
import { useParams, useNavigate } from "react-router-dom";

import StackThumbnail from "./StackThumbnail.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import SuggestionBadge from "./suggestions/SuggestionBadge.jsx";
import QuarantineToggle from "./QuarantineToggle.jsx";
import ShoppingCartMenu from "./ShoppingCartMenu.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import ViewPills from './ViewPills.jsx';
import { buildUrl, inheritView } from './urls.js';


export default function TargetPanel() {
	const { stacks, targets, isQuarantined } = useContext(AstroFlowContext);

	const params = useParams();
	const navigate = useNavigate();

	const target = targets[params.target_id];

	const hierarchy = [];
	if (params.mosaic_id) hierarchy.push({ type: 'mosaic', id: params.mosaic_id });
	hierarchy.push({ type: 'target', id: params.target_id });

	const view = (params.pipeline && params.display && params.filter)
		? { pipeline: params.pipeline, display: params.display, filter: params.filter }
		: null;

	const handleViewChange = (nextView) => {
		navigate(buildUrl({ hierarchy, view: nextView }));
	};

	// When a view is picked, drilldown URLs propagate it (stack defaults
	// for display via inheritView); without a view, links degrade to the
	// hierarchy-only URL so the stack page can pick its own defaults.
	const handleStackClick = (stack_id) => (event) => {
		const childHierarchy = [...hierarchy, { type: 'stack', id: stack_id }];
		const url = view
			? buildUrl({ hierarchy: childHierarchy, view: inheritView(view, 'stack') })
			: buildUrl({ hierarchy: childHierarchy });
		window.open(url, '_blank');
	};

	return (
		<div>
			<div style={{ display: "flex", alignItems: "center", marginLeft: "2em", marginBottom: "0.5em" }}>
				<ShoppingCartMenu objectType="target" objectId={params.target_id} />
				<QuarantineToggle objectType="target" objectId={params.target_id} />
				<SuggestionBadge objectType="target" objectId={params.target_id} />
			</div>
			<ViewPills hierarchy={hierarchy} value={view} onChange={handleViewChange} />
			<div style={{
					display: "grid",
					gridTemplateColumns: "repeat(4, 1fr)",
					maxHeight: "80vh",
					gap: "1em",
					marginLeft: "2em",
					marginRight: "2em",
					opacity: isQuarantined("target", params.target_id) ? 0.5 : 1,
				}}>
					{target?.stack_ids?.map((stack_id) => (
						<StackThumbnail
							key={"cell_" + stack_id}
							stacks={stacks}
							stack_id={stack_id}
							handleClick={handleStackClick(stack_id)}
						>
							<DecisionBadge objectType="stack" objectId={stack_id} />
							<ConcernsBadge objectType="stack" objectId={stack_id} />
						</StackThumbnail>
					))}
			</div>
		</div>
	);
}
