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

const StateChip = ({ state }) => {
  // Map your real states here (SUCCEEDED / FAILED / SKIPPED / RUNNING / etc.)
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

export function NodeStatusDialog({ open, onClose, nodeId, nodeLabel }) {
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [details, setDetails] = useState(null);

  // Suggested payload shape returned by your backend:
  // {
  //   id, label,
  //   state: "SUCCEEDED"|"FAILED"|"SKIPPED"|...,
  //   open_image_url?: "/api/open_image?hash=...",
  //   image_url?: "/tiles/..." or "/api/image/..." (optional if you want preview)
  //   exception?: { message?: string, type?: string, stack?: string },
  //   log?: string
  // }
  useEffect(() => {
    if (!open || !nodeId) return;
    let cancelled = false;

    setLoading(true);
    setErr(null);
    setDetails(null);

    (async () => {
      try {
        const res = await fetch(`/api/node_status/${nodeId}`);
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

  const title = details?.label ?? nodeLabel ?? details?.id ?? nodeId ?? "Node";

  const state = details?.state ?? null;
  const hasImage = Boolean(details?.open_image_url);
  const hasException = Boolean(details?.exception?.stack || details?.exception?.message);
  const hasLog = Boolean(details?.log);

  const exceptionSummary = useMemo(() => {
    if (!details?.exception) return "";
    const t = details.exception.type ? `${details.exception.type}: ` : "";
    return `${t}${details.exception.message ?? ""}`.trim();
  }, [details]);

  const openImage = useCallback(() => {
    const u = details?.open_image_url;
    if (!u) return;
    // If your server endpoint triggers native OS open, a normal navigation is fine.
    // If you prefer not to navigate away, do window.open(u, "_blank") instead.
    window.location.href = u;
  }, [details]);

  const copyException = useCallback(() => {
    copyToClipboard(details?.exception?.stack ?? details?.exception?.message ?? "");
  }, [details]);

  const copyLog = useCallback(() => {
    copyToClipboard(details?.log ?? "");
  }, [details]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth={false} fullWidth>
      <DialogTitle sx={{ pr: 6 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="h6" noWrap>{title}</Typography>
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
          {/* Top summary / actions */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack direction="row" spacing={1.5} alignItems="center" justifyContent="space-between">
              <Stack direction="row" spacing={1} alignItems="center">
                {hasException && <ErrorOutlineIcon fontSize="small" />}
                <Typography variant="body2" color="text.secondary">
                  {hasException ? "This node raised an exception." : "No exception reported."}
                </Typography>
              </Stack>

              <Stack direction="row" spacing={1} alignItems="center">
                {hasImage && (
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
                  <Typography variant="body2" color="text.secondary">Loading…</Typography>
                </Stack>
              </>
            )}
          </Paper>

          {/* Scrollable lower area */}
          <Box sx={{ flex: 1, overflow: "auto" }}>
            <Stack spacing={1.5}>
              {/* Exception accordion */}
              <Accordion disableGutters defaultExpanded={false} sx={{ border: 1, borderColor: "divider" }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
                    <ErrorOutlineIcon fontSize="small" />
                    <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }} noWrap>
                      Exception
                    </Typography>
                    {hasException ? (
                      <Typography variant="caption" color="text.secondary" noWrap sx={{ maxWidth: 520 }}>
                        {exceptionSummary}
                      </Typography>
                    ) : (
                      <Typography variant="caption" color="text.secondary">None</Typography>
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
                      {details?.exception?.stack ??
                        details?.exception?.message ??
                        "No exception information."}
                    </Typography>
                  </Paper>
                </AccordionDetails>
              </Accordion>

              {/* Log accordion */}
              <Accordion disableGutters defaultExpanded={false} sx={{ border: 1, borderColor: "divider" }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
                    <ArticleIcon fontSize="small" />
                    <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }} noWrap>
                      Log
                    </Typography>
                    {hasLog ? (
                      <Typography variant="caption" color="text.secondary">
                        {`${String(details.log).split("\n").length} lines`}
                      </Typography>
                    ) : (
                      <Typography variant="caption" color="text.secondary">None</Typography>
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
                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: "background.default" }}>
                    <Typography component="pre" sx={monospaceSx}>
                      {details?.log ?? "No log output."}
                    </Typography>
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
