import React, { useContext } from "react";
import { useParams } from "react-router-dom";
import { Box } from "@mui/material";

import StackThumbnail from "./StackThumbnail.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import SuggestionBadge from "./suggestions/SuggestionBadge.jsx";
import QuarantineToggle from "./QuarantineToggle.jsx";
import ShoppingCartMenu from "./ShoppingCartMenu.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import { PipelineSection } from './MosaicDashboard.jsx';
import { useVersionsMatrix } from './useVersions.js';
import { buildUrl, inheritView } from './urls.js';


export default function TargetPanel() {
	const { stacks, targets, isQuarantined, computeGraphStatus } = useContext(AstroFlowContext);

	const params = useParams();

	const target = targets[params.target_id];

	const hierarchy = [];
	if (params.mosaic_id) hierarchy.push({ type: 'mosaic', id: params.mosaic_id });
	hierarchy.push({ type: 'target', id: params.target_id });

	const view = (params.pipeline && params.display && params.filter)
		? { pipeline: params.pipeline, display: params.display, filter: params.filter }
		: null;

	// Pipeline output sections — same per-pipeline tap matrix the mosaic
	// dashboard shows. Empty (no sections) for a target no pipeline has
	// produced taps for.
	const pipelineOutputs = useVersionsMatrix(
		"target", params.target_id, computeGraphStatus);
	const pipelines = (pipelineOutputs?.pipelines || [])
		.filter((p) => (p.taps || []).length > 0);

	// No page-level hero here either: each PipelineSection renders its own
	// canonical preview, deep-linked into that pipeline's viewer.

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
			{pipelines.length > 0 && (
				<Box sx={{ mx: "2em", mt: "1.5em" }}>
					{pipelines.map((p) => (
						<PipelineSection
							key={p.name}
							objectType="target"
							objectId={params.target_id}
							pipeline={p}
						/>
					))}
				</Box>
			)}
		</div>
	);
}
