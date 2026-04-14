// MosaicViewer.jsx — the detail page for a single
// (mosaic × implementation × quality × filter). The dashboard
// (MosaicDashboard.jsx) is the entry point; this is what you see
// after clicking a thumbnail.
import React, { useContext, useEffect, useMemo, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";

import {
	Box,
	Breadcrumbs,
	Button,
	Divider,
	IconButton,
	Link,
	Paper,
	Tab,
	Tabs,
	Tooltip,
	Typography,
} from "@mui/material";

import FullscreenIcon from "@mui/icons-material/Fullscreen";
import ShareIcon from "@mui/icons-material/Share";

import AstroFlowContext from "./AstroFlowContext.jsx";
import { AstroFlowImage } from "./AstroFlowImage.jsx";
import TelescopiusDialog from "./TelescopiusDialog.jsx";
import ToggleIconButton from "./ToggleIconButton";
import WorkflowGraph from "./WorkflowGraph.jsx";
import darkTheme from "./theme.js";

const QUALITY_ORDER = ["PostProcessed", "Master", "DownscaledMaster", "Subframe"];


function AlternateVersionsStrip({ mosaicId, implementation, quality, filter, versions, filters }) {
    const navigate = useNavigate();
    const go = (impl, q, f) => navigate(`/mosaic/${mosaicId}/${impl}/${q}/${f}`);

    // Unique implementations that have any version.
    const implementations = [...new Set(versions.map((v) => v.implementation))];
    const qualities = QUALITY_ORDER.filter((q) =>
        versions.some((v) => v.quality === q && v.implementation === implementation && v.exists)
    );

    const exists = (impl, q, f) => versions.some(
        (v) => v.implementation === impl && v.quality === q && v.filter === f && v.exists
    );

    const Pill = ({ active, disabled, onClick, children }) => (
        <Button
            size="small"
            variant={active ? "contained" : "outlined"}
            disabled={disabled}
            onClick={onClick}
            sx={{ minWidth: 0, mr: 0.5, py: 0.25 }}
        >
            {children}
        </Button>
    );

    return (
        <Box sx={{
            display: 'flex', gap: 2, alignItems: 'center',
            px: 2, py: 1,
            borderBottom: '1px solid', borderColor: 'divider',
            flexWrap: 'wrap',
        }}>
            <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', mr: 1 }}>
                    Pipeline:
                </Typography>
                {implementations.map((impl) => (
                    <Pill
                        key={impl}
                        active={impl === implementation}
                        disabled={!exists(impl, quality, filter)}
                        onClick={() => go(impl, quality, filter)}
                    >
                        {impl}
                    </Pill>
                ))}
            </Box>
            <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', mr: 1 }}>
                    Quality:
                </Typography>
                {qualities.map((q) => (
                    <Pill
                        key={q}
                        active={q === quality}
                        disabled={!exists(implementation, q, filter)}
                        onClick={() => go(implementation, q, filter)}
                    >
                        {q}
                    </Pill>
                ))}
            </Box>
            <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', mr: 1 }}>
                    Filter:
                </Typography>
                {filters.map((f) => (
                    <Pill
                        key={f}
                        active={f === filter}
                        disabled={!exists(implementation, quality, f)}
                        onClick={() => go(implementation, quality, f)}
                    >
                        {f}
                    </Pill>
                ))}
            </Box>
        </Box>
    );
}


export default function MosaicViewer() {
    const { mosaic_id, implementation, quality, filter } = useParams();
    const navigate = useNavigate();
    const { stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

	const [tabIndex, setTabIndex] = useState(0);
	const [showOutline, setShowOutline] = useState(true);
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

			{/* Breadcrumb */}
			<Box sx={{ px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
				<Breadcrumbs>
					<Link
						component="button"
						underline="hover"
						color="inherit"
						onClick={() => navigate(`/mosaic/${mosaic_id}`)}
					>
						{mosaic.name || `Mosaic ${mosaic_id.slice(0, 8)}`}
					</Link>
					<Typography color="text.primary">{implementation}</Typography>
					<Typography color="text.primary">{quality}</Typography>
					<Typography color="text.primary">{filter}</Typography>
				</Breadcrumbs>
			</Box>

			{/* Alternate versions strip */}
			<AlternateVersionsStrip
				mosaicId={mosaic_id}
				implementation={implementation}
				quality={quality}
				filter={filter}
				versions={versionsData.versions}
				filters={versionsData.filters}
			/>

			{/* Control bar */}
			<Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 2, py: 1 }}>
				<ToggleIconButton
					name="outline"
					icon={<FullscreenIcon />}
					checked={showOutline}
					handleChange={() => setShowOutline((v) => !v)}
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
