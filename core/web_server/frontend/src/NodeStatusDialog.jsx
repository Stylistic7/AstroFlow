import React, { useEffect, useMemo, useState, useCallback } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Box,
  Typography,
  Chip,
  Button,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Paper,
  Divider,
  Tooltip,
  CircularProgress,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ImageIcon from "@mui/icons-material/Image";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import ArticleIcon from "@mui/icons-material/Article";

const monospaceSx = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 12,
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  lineHeight: 1.35,
};

const levelColors = {
  debug: "text.disabled",
  info: "text.primary",
  warning: "warning.main",
  error: "error.main",
  critical: "error.dark",
};

const levelWeight = {
  debug: 400,
  info: 400,
  warning: 600,
  error: 600,
  critical: 700,
};

const StructuredLogEntry = ({ entry }) => {
  // entry is either a structlog dict or a plain string
  if (typeof entry === "string") {
    return <Typography sx={monospaceSx}>{entry}</Typography>;
  }

  const level = entry.level ?? "info";
  const event = entry.event ?? "";
  const timestamp = entry.timestamp ?? "";
  const color = levelColors[level] ?? "text.primary";
  const weight = levelWeight[level] ?? 400;

  // Collect extra keys (everything except event, level, timestamp)
  const extras = Object.entries(entry).filter(
    ([k]) => !["event", "level", "timestamp"].includes(k)
  );

  return (
    <Box sx={{ display: "flex", gap: 1, alignItems: "baseline", py: 0.25 }}>
      {timestamp && (
        <Typography sx={{ ...monospaceSx, color: "text.disabled", flexShrink: 0 }}>
          {timestamp.split("T")[1]?.split(".")[0] ?? timestamp}
        </Typography>
      )}
      <Chip
        label={level.toUpperCase()}
        size="small"
        sx={{
          height: 18,
          fontSize: 10,
          fontWeight: 600,
          bgcolor: level === "debug" ? "action.hover" :
                   level === "warning" ? "warning.light" :
                   level === "error" || level === "critical" ? "error.light" :
                   "info.light",
          color: level === "error" || level === "critical" ? "error.contrastText" :
                 level === "warning" ? "warning.contrastText" :
                 "text.primary",
        }}
      />
      <Typography sx={{ ...monospaceSx, color, fontWeight: weight, flex: 1 }}>
        {event}
        {extras.length > 0 && (
          <Typography component="span" sx={{ ...monospaceSx, color: "text.secondary" }}>
            {" "}
            {extras.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ")}
          </Typography>
        )}
      </Typography>
    </Box>
  );
};

const StateChip = ({ state }) => {
  const { label, color, variant } = (() => {
    switch (state) {
      case "SUCCEEDED":
      case "SUCCESS":
        return { label: "Succeeded", color: "success", variant: "filled" };
      case "FAILED":
        return { label: "Failed", color: "error", variant: "filled" };
      case "SKIPPED":
        return { label: "Skipped", color: "default", variant: "outlined" };
      case "RUNNING":
        return { label: "Running", color: "info", variant: "filled" };
      default:
        return { label: state ?? "Unknown", color: "default", variant: "outlined" };
    }
  })();
  return <Chip size="small" label={label} color={color} variant={variant} />;
};

const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text ?? "");
  } catch {
    // ignore
  }
};

/**
 * Expected backend payload:
 * {
 *   result_type: string | null,
 *   logs: any[],
 *   exception: string | null,
 *   stack_trace: string | null,
 *   path?: string | null,                 // if present -> show "Open result"
 *   open_image_url?: string | null,       // optional legacy hook
 * }
 */
export function NodeStatusDialog({ open, onClose, nodeId, nodeLabel }) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [details, setDetails] = useState(null);

  // UI status for "open result" action
  const [openResultBusy, setOpenResultBusy] = useState(false);
  const [openResultErr, setOpenResultErr] = useState(null);

  useEffect(() => {
    if (!open || !nodeId) return;
    let cancelled = false;

    setLoading(true);
    setErr(null);
    setDetails(null);
    setOpenResultBusy(false);
    setOpenResultErr(null);

    (async () => {
      try {
        const res = await fetch(`/api/node_status/${encodeURIComponent(nodeId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) setDetails(data);
      } catch (e) {
        if (!cancelled) setErr(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [open, nodeId]);

  const resultType = details?.result_type ?? null;

  const exceptionMessage = details?.exception ?? "";
  const stackTrace = details?.stack_trace ?? "";

  const logsArr = Array.isArray(details?.logs) ? details.logs : [];
  const logText = useMemo(() => {
    return logsArr.map(entry =>
      typeof entry === "string" ? entry :
      `${entry.timestamp ?? ""} [${(entry.level ?? "info").toUpperCase()}] ${entry.event ?? ""}` +
      Object.entries(entry)
        .filter(([k]) => !["event", "level", "timestamp"].includes(k))
        .map(([k, v]) => ` ${k}=${JSON.stringify(v)}`)
        .join("")
    ).join("\n");
  }, [logsArr]);

  const hasException = Boolean(stackTrace || exceptionMessage);
  const hasLog = logText.length > 0;

  const state = useMemo(() => {
    if (!details) return null;
    if (hasException || resultType) return "FAILED";
    return "SUCCEEDED";
  }, [details, hasException, resultType]);

  const openImageUrl = details?.open_image_url ?? null;
  const hasImage = Boolean(openImageUrl);

  const resultPath = details?.path ?? null;
  const hasPath = Boolean(resultPath);

  const title = nodeLabel ?? nodeId ?? "Node";

  const exceptionSummary = useMemo(() => {
    if (!hasException && !resultType) return "";
    const t = resultType ? `${resultType}: ` : "";
    return `${t}${exceptionMessage}`.trim();
  }, [hasException, resultType, exceptionMessage]);

  const openImage = useCallback(() => {
    if (!openImageUrl) return;
    // POST: the backend rejects GET so a hostile page can't cross-origin
    // trigger a PixInsight launch by navigation.
    fetch(openImageUrl, { method: "POST" }).catch(console.error);
  }, [openImageUrl]);

  const openResultPath = useCallback(async () => {
    if (!nodeId || !hasPath) return;

    setOpenResultBusy(true);
    setOpenResultErr(null);

    try {
      const res = await fetch(
        `/api/open_node_result_path/${encodeURIComponent(nodeId)}`
      ); // GET by default

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      // Optionally consume body
      try {
        await res.text();
      } catch {
        // ignore
      }
    } catch (e) {
      setOpenResultErr(String(e));
    } finally {
      setOpenResultBusy(false);
    }
  }, [nodeId, hasPath]);
  
  const copyException = useCallback(() => {
    copyToClipboard(stackTrace || exceptionMessage || "");
  }, [stackTrace, exceptionMessage]);

  const copyLog = useCallback(() => {
    copyToClipboard(logText);
  }, [logText]);

  const logLineCount = useMemo(() => (hasLog ? logText.split("\n").length : 0), [hasLog, logText]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth={false} fullWidth>
      <DialogTitle sx={{ pr: 6 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="h6" noWrap>
              {title}
            </Typography>
            <Typography variant="caption" color="text.secondary" noWrap>
              {nodeId ? `id: ${nodeId}` : ""}
            </Typography>
          </Box>

          {state && <StateChip state={state} />}

          <IconButton
            size="small"
            onClick={onClose}
            sx={{ position: "absolute", right: 12, top: 12 }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ height: 540, overflow: "hidden" }}>
        <Box sx={{ display: "flex", flexDirection: "column", height: "100%", gap: 2 }}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack direction="row" spacing={1.5} alignItems="center" justifyContent="space-between">
              <Stack direction="row" spacing={1} alignItems="center">
                {hasException && <ErrorOutlineIcon fontSize="small" />}
                <Typography variant="body2" color="text.secondary">
                  {hasException ? "This node raised an exception." : "No exception reported."}
                </Typography>
              </Stack>

              <Stack direction="row" spacing={1} alignItems="center">
                {hasPath && (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<ImageIcon />}
                    onClick={openResultPath}
                    disabled={openResultBusy}
                  >
                    {openResultBusy ? "Opening…" : "Open result"}
                  </Button>
                )}

                {!hasPath && hasImage && (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<ImageIcon />}
                    onClick={openImage}
                  >
                    Open image
                  </Button>
                )}
              </Stack>
            </Stack>

            {hasPath && (
              <>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                  Result path
                </Typography>
                <Typography sx={monospaceSx}>{String(resultPath)}</Typography>
              </>
            )}

            {openResultErr && (
              <>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="body2" color="error">
                  Failed to open result: {openResultErr}
                </Typography>
              </>
            )}

            {err && (
              <>
                <Divider sx={{ my: 1.5 }} />
                <Typography variant="body2" color="error">
                  Failed to load details: {err}
                </Typography>
              </>
            )}

            {loading && (
              <>
                <Divider sx={{ my: 1.5 }} />
                <Stack direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Loading…
                  </Typography>
                </Stack>
              </>
            )}
          </Paper>

          <Box sx={{ flex: 1, overflow: "auto" }}>
            <Stack spacing={1.5}>
              <Accordion
                disableGutters
                defaultExpanded={false}
                sx={{ border: 1, borderColor: "divider" }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
                    <ErrorOutlineIcon fontSize="small" />
                    <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }} noWrap>
                      Exception
                    </Typography>
                    {hasException ? (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        noWrap
                        sx={{ maxWidth: 520 }}
                      >
                        {exceptionSummary}
                      </Typography>
                    ) : (
                      <Typography variant="caption" color="text.secondary">
                        None
                      </Typography>
                    )}
                  </Stack>
                </AccordionSummary>

                <AccordionDetails>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                    <Tooltip title="Copy">
                      <IconButton size="small" onClick={copyException} disabled={!hasException}>
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Typography variant="caption" color="text.secondary">
                      Stack trace
                    </Typography>
                  </Stack>

                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: "background.default" }}>
                    <Typography component="pre" sx={monospaceSx}>
                      {stackTrace || exceptionMessage || "No exception information."}
                    </Typography>
                  </Paper>
                </AccordionDetails>
              </Accordion>

              <Accordion
                disableGutters
                defaultExpanded={false}
                sx={{ border: 1, borderColor: "divider" }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
                    <ArticleIcon fontSize="small" />
                    <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }} noWrap>
                      Log
                    </Typography>
                    {hasLog ? (
                      <Typography variant="caption" color="text.secondary">
                        {`${logLineCount} lines`}
                      </Typography>
                    ) : (
                      <Typography variant="caption" color="text.secondary">
                        None
                      </Typography>
                    )}
                  </Stack>
                </AccordionSummary>

                <AccordionDetails>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                    <Tooltip title="Copy">
                      <IconButton size="small" onClick={copyLog} disabled={!hasLog}>
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Typography variant="caption" color="text.secondary">
                      Node output
                    </Typography>
                  </Stack>

                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: "background.default", maxHeight: 300, overflow: "auto" }}>
                    {hasLog ? (
                      logsArr.map((entry, i) => <StructuredLogEntry key={i} entry={entry} />)
                    ) : (
                      <Typography sx={monospaceSx} color="text.secondary">
                        No log output.
                      </Typography>
                    )}
                  </Paper>
                </AccordionDetails>
              </Accordion>
            </Stack>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}