// MosaicViewer.jsx — the detail page for a single
// (mosaic × implementation × quality × filter). The dashboard
// (MosaicDashboard.jsx) is the entry point; this is what you see
// after clicking a thumbnail.
import React, { useContext, useEffect, useMemo, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";

import {
	Box,
	Divider,
	IconButton,
	Paper,
	Tab,
	Tabs,
	Tooltip,
	Typography,
} from "@mui/material";

import FullscreenIcon from "@mui/icons-material/Fullscreen";
import ShareIcon from "@mui/icons-material/Share";
import PinIcon from "@mui/icons-material/Pin";
import LaunchIcon from "@mui/icons-material/Launch";

import AstroFlowContext from "./AstroFlowContext.jsx";
import { AstroFlowImage } from "./AstroFlowImage.jsx";
import TelescopiusDialog from "./TelescopiusDialog.jsx";
import ToggleIconButton from "./ToggleIconButton";
import WorkflowGraph from "./WorkflowGraph.jsx";
import darkTheme from "./theme.js";

const QUALITY_ORDER = ["PostProcessed", "Master", "DownscaledMaster", "Subframe"];


export default function MosaicViewer() {
    const { mosaic_id, implementation, quality, filter } = useParams();
    const navigate = useNavigate();
    const { stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

	const [tabIndex, setTabIndex] = useState(0);
	const [showOutline, setShowOutline] = useState(true);
	const [showPanelNumbers, setShowPanelNumbers] = useState(false);
	const [telescopiusOpen, setTelescopiusOpen] = useState(false);
	const [versionsData, setVersionsData] = useState({ filters: [], versions: [] });

    const mosaic = mosaics?.[mosaic_id];

    useEffect(() => {
        fetch(`/api/mosaic_versions/${mosaic_id}`)
            .then((res) => res.ok ? res.json() : null)
            .then((data) => data && setVersionsData(data))
            .catch((e) => console.warn(e));
    }, [mosaic_id, computeGraphStatus]);

    // Keyboard shortcuts to flip between versions while staying at full screen:
    //   [ ]   cycle implementation
    //   { }   cycle quality
    //   , .   cycle filter
    useEffect(() => {
        const handler = (e) => {
            // Ignore when typing in an input.
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

            const { implementations, qualities, filters } = (() => {
                const vs = versionsData.versions;
                const impls = [...new Set(vs.map((v) => v.implementation))];
                const quals = QUALITY_ORDER.filter((q) =>
                    vs.some((v) => v.quality === q && v.implementation === implementation && v.exists)
                );
                return { implementations: impls, qualities: quals, filters: versionsData.filters };
            })();

            const exists = (impl, q, f) => versionsData.versions.some(
                (v) => v.implementation === impl && v.quality === q && v.filter === f && v.exists
            );

            const cycle = (list, current, dir) => {
                const idx = list.indexOf(current);
                if (idx < 0) return current;
                return list[(idx + dir + list.length) % list.length];
            };
            const cycleToExisting = (list, current, dir, predicate) => {
                // Step through until we find a combination that exists.
                let candidate = current;
                for (let i = 0; i < list.length; i++) {
                    candidate = cycle(list, candidate, dir);
                    if (predicate(candidate)) return candidate;
                }
                return current;
            };

            if (e.key === "[" || e.key === "]") {
                const dir = e.key === "]" ? 1 : -1;
                const next = cycleToExisting(
                    implementations, implementation, dir,
                    (i) => exists(i, quality, filter),
                );
                if (next !== implementation) navigate(`/mosaic/${mosaic_id}/${next}/${quality}/${filter}`);
                e.preventDefault();
            } else if (e.key === "{" || e.key === "}") {
                const dir = e.key === "}" ? 1 : -1;
                const next = cycleToExisting(
                    qualities, quality, dir,
                    (q) => exists(implementation, q, filter),
                );
                if (next !== quality) navigate(`/mosaic/${mosaic_id}/${implementation}/${next}/${filter}`);
                e.preventDefault();
            } else if (e.key === "," || e.key === ".") {
                const dir = e.key === "." ? 1 : -1;
                const next = cycleToExisting(
                    filters, filter, dir,
                    (f) => exists(implementation, quality, f),
                );
                if (next !== filter) navigate(`/mosaic/${mosaic_id}/${implementation}/${quality}/${next}`);
                e.preventDefault();
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [mosaic_id, implementation, quality, filter, versionsData, navigate]);

    // Furthest-processed file exists for the current (impl, filter)
    // when versionsData has a Master version flagged exists. Backend
    // accepts either the canonical store output or the per-filter
    // merge-progress TIFF, both of which set exists=true.
    const masterExists = versionsData.versions.some(
        (v) => v.implementation === implementation
            && v.quality === "Master"
            && v.filter === filter
            && v.exists,
    );

    const openInPixInsight = () => {
        fetch(`/api/open_mosaic_master/${mosaic_id}/${implementation}/${filter}`)
            .then((r) => { if (!r.ok) console.warn("open_mosaic_master failed", r.status); })
            .catch(console.error);
    };

    if (!mosaic) {
        return <Typography sx={{ p: 3 }}>Loading…</Typography>;
    }

	return (
		<Paper
			elevation={2}
			sx={{
				borderRadius: 0,
				borderTop: "1px solid",
				borderColor: "divider",
				display: "flex",
				flexDirection: "column",
				flex: 1,
				minHeight: 0,
			}}
		>
			<TelescopiusDialog
				open={telescopiusOpen}
				onClose={() => setTelescopiusOpen(false)}
				stacks={stacks}
				targets={targets}
				mosaic={mosaic}
			/>

			{/* Implementation / quality / filter switching is now handled
			    by the pill crumbs in AstroFlowBreadcrumbs. */}

			{/* Control bar */}
			<Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 2, py: 1 }}>
				<ToggleIconButton
					name="outline"
					icon={<FullscreenIcon />}
					checked={showOutline}
					handleChange={() => setShowOutline((v) => !v)}
				/>
				<ToggleIconButton
					name="panel numbers"
					icon={<PinIcon />}
					checked={showPanelNumbers}
					handleChange={() => setShowPanelNumbers((v) => !v)}
				/>
				<Tooltip title="Export to Telescopius">
					<IconButton
						size="small"
						onClick={() => setTelescopiusOpen(true)}
						sx={{ color: 'text.secondary' }}
					>
						<ShareIcon fontSize="small" />
					</IconButton>
				</Tooltip>
				<Tooltip title={masterExists ? "Open in PixInsight" : "No master image available yet"}>
					<span>
						<IconButton
							size="small"
							disabled={!masterExists}
							onClick={openInPixInsight}
							sx={{ color: 'text.secondary' }}
						>
							<LaunchIcon fontSize="small" />
						</IconButton>
					</span>
				</Tooltip>
			</Box>

			{/* Tabs */}
			<Box sx={{ backgroundColor: "rgba(255,255,255,0.05)", px: 2 }}>
				<Tabs
					value={tabIndex}
					onChange={(_e, newIndex) => setTabIndex(newIndex)}
					indicatorColor="primary"
					textColor="primary"
					variant="scrollable"
					scrollButtons="auto"
				>
					<Tab label="Merged" />
					<Tab label="Graph" />
				</Tabs>
			</Box>
			<Divider />

			{/* Tab content */}
			<Box sx={{ flex: 1, p: 3, minHeight: 0, position: "relative" }}>
				<Box
					hidden={tabIndex !== 0}
					sx={{
						height: "100%",
						overflow: "hidden",
						position: "absolute",
						inset: 0,
						p: 3,
					}}
				>
					<AstroFlowImage
						objectType="mosaic"
						objectId={mosaic_id}
						filter={filter}
						showPolygons={showOutline}
						showPanelNumbers={showPanelNumbers}
						qualityLevel={quality}
						implementation={implementation}
						showQualitySelector={false}
					/>
				</Box>

				<Box
					hidden={tabIndex !== 1}
					sx={{
						height: "100%",
						overflow: "auto",
						position: "absolute",
						inset: 0,
						p: 3,
					}}
				>
					<WorkflowGraph
						theme={darkTheme}
						url={`/api/workflow_graph/mosaic/${mosaic_id}/${filter}/${quality}`}
					/>
				</Box>
			</Box>
		</Paper>
	);
}
