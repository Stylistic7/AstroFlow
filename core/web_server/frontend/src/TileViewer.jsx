import { useEffect, useRef, useState, useLayoutEffect, useContext } from 'react';
import * as PIXI from 'pixi.js';
import { Stage, useApp } from '@pixi/react';
import { useParams, useNavigate } from 'react-router-dom';
import { Viewport } from 'pixi-viewport';

import AstroFlowContext from './AstroFlowContext.jsx';


const TILE_SIZE = 256;

const TileViewer = ({ objectType: objectTypeProp, imageHash: imageHashProp, filter: filterProp, qualityLevel: qualityLevelProp, showPolygons }) => {
  const { objectType: objectTypeParam, imageHash: imageHashParam, filter: filterParam } = useParams();

  const objectType = objectTypeProp ?? objectTypeParam;
  const imageHash = imageHashProp ?? imageHashParam;
  const filter = filterProp ?? filterParam;

  const [metadata, setMetadata] = useState(null);
  const [tileMetadata, setTileMetadata] = useState(null);
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  const { mosaics, stacks, targets, showQuarantined, isQuarantined, computeGraphStatus } = useContext(AstroFlowContext);

  const navigate = useNavigate();

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(() => {
      const bounds = containerRef.current.getBoundingClientRect();
      setContainerSize({ width: bounds.width, height: bounds.height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (qualityLevelProp) {
      // Quality level provided by parent — fetch metadata directly.
      const metaUrl = filter !== undefined
        ? `/api/metadata/${objectType}/${imageHash}/${filter}/${qualityLevelProp}`
        : `/api/metadata/${objectType}/${imageHash}/${qualityLevelProp}`;
      fetch(metaUrl)
        .then((res) => res.ok ? res.json() : null)
        .then((meta) => { if (meta) setMetadata(meta); })
        .catch((e) => console.warn(e));
    } else {
      // Auto-detect best quality level.
      const bestUrl = filter !== undefined
        ? `/api/available_qualities/${objectType}/${imageHash}/${filter}`
        : `/api/available_qualities/${objectType}/${imageHash}`;
      fetch(bestUrl)
        .then((res) => res.ok ? res.json() : null)
        .then((best) => {
          if (!best || !best.quality_level) return;
          const level = best.quality_level;
          const metaUrl = filter !== undefined
            ? `/api/metadata/${objectType}/${imageHash}/${filter}/${level}`
            : `/api/metadata/${objectType}/${imageHash}/${level}`;
          return fetch(metaUrl).then((res) => res.json());
        })
        .then((meta) => { if (meta) setMetadata(meta); })
        .catch((e) => console.warn(e));
    }
  }, [objectType, imageHash, filter, qualityLevelProp]);

  useEffect(() => {
    const level = qualityLevelProp || 'auto';
    fetch(`/tile_metadata/${objectType}/${imageHash}/${filter}/${level}`)
      .then((res) => res.ok ? res.json() : null)
      .then((meta) => { if (meta) setTileMetadata(meta); });
  }, [objectType, imageHash, filter, qualityLevelProp, computeGraphStatus]);

  const onClick = (poly, event) => {
    const e = event.nativeEvent; // underlying DOM PointerEvent

    let url;
    if (poly.type === 'stack' && poly.target_id) {
      // Stack polygon inside a mosaic/target — build the full waterfall path.
      url = `target/${poly.target_id}/stack/${poly.stable_hash}`;
    } else if (poly.type === 'target' && filter && targets?.[poly.stable_hash]) {
      const target = targets[poly.stable_hash];
      const stackId = target.stack_ids?.find(s => stacks?.[s]?.filter === filter);
      url = stackId
        ? `target/${poly.stable_hash}/stack/${stackId}`
        : `target/${poly.stable_hash}`;
    } else {
      url = `${poly.type}/${poly.stable_hash}`;
    }

    if (e.metaKey || e.ctrlKey) {
      window.open(`${window.location.pathname}/${url}`, '_blank');
    } else {
      navigate(url);
    }
  };

  if (!metadata || !tileMetadata || !containerRef.current) return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;

  const worldWidth = tileMetadata.tileWidth * Math.pow(2, tileMetadata.maxZoom);
  const worldHeight = tileMetadata.tileHeight * Math.pow(2, tileMetadata.maxZoom);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
      <Stage
        width={containerSize.width}
        height={containerSize.height}
        options={{
          backgroundColor: 0x000000,
          resizeTo: containerRef.current,
        }}
        style={{ width: '100%', height: '100%' }}
      >
        <ImperativeViewport
          objectType={objectType}
          imageHash={imageHash}
          filter={filter}
          qualityLevel={qualityLevelProp}
          port={tileMetadata.port}
          imageWidth={tileMetadata.width}
          imageHeight={tileMetadata.height}
          worldWidth={worldWidth}
          worldHeight={worldHeight}
          maxZoom={tileMetadata.maxZoom}
          mtime={tileMetadata.mtime}
          showPolygons={showPolygons}
          metadata={metadata}
          mosaics={mosaics}
          onClick={onClick}
          showQuarantined={showQuarantined}
          isQuarantined={isQuarantined}
          stacks={stacks}
        />
      </Stage>
    </div>
  );
};


/** Draw an exposure-duration clock face — same dial style as
    ExposureDurationWidget — into a PIXI Graphics, centered at (cx, cy)
    with the given pixel radius. `hours` is clamped to [0, 12). */
function drawExposureClock(g, cx, cy, radius, hours) {
  // Background disc.
  g.beginFill(0x333333);
  g.drawCircle(cx, cy, radius);
  g.endFill();

  // Wedge for the integration. ExposureDurationWidget paints the same
  // value twice (red then blue) so the visible colour is blue.
  const fraction = Math.min(Math.max(hours, 0), 11.999) / 12;
  if (fraction > 0) {
    const start = -Math.PI / 2;
    const end = start + 2 * Math.PI * fraction;
    g.beginFill(0x4278f5);
    g.moveTo(cx, cy);
    g.arc(cx, cy, radius, start, end);
    g.lineTo(cx, cy);
    g.endFill();
  }

  // 12 tick marks.
  g.lineStyle(radius * 0.05, 0xffffff, 1);
  for (let h = 0; h < 12; h++) {
    const theta = (2 * Math.PI * h / 12) - Math.PI / 2;
    g.moveTo(cx + radius * 0.75 * Math.cos(theta), cy + radius * 0.75 * Math.sin(theta));
    g.lineTo(cx + radius * Math.cos(theta), cy + radius * Math.sin(theta));
  }

  // Outer ring.
  g.lineStyle(radius * 0.18, 0xffffff, 1);
  g.drawCircle(cx, cy, radius);
  g.lineStyle(0);
}


const ImperativeViewport = ({ objectType, imageHash, filter, qualityLevel, port, imageWidth, imageHeight, worldWidth, worldHeight, maxZoom, mtime, showPolygons, metadata, mosaics, onClick, showQuarantined, isQuarantined, stacks }) => {
  const app = useApp();
  const viewportRef = useRef(null);
  const spritesRef = useRef({});

  // When the tile pyramid file changes (new merge completed), clear
  // the sprite cache so tiles are re-fetched with the new content.
  useEffect(() => {
    if (!viewportRef.current) return;
    Object.values(spritesRef.current).forEach(s => {
      s.parent?.removeChild(s);
      s.destroy();
    });
    spritesRef.current = {};
  }, [mtime]);

  useEffect(() => {
    const screenWidth = app.renderer.view.clientWidth;
    const screenHeight = app.renderer.view.clientHeight;

    const viewport = new Viewport({
      screenWidth,
      screenHeight,
      worldWidth,
      worldHeight,
      events: app.renderer.events,
      interaction: app.renderer.plugins.interaction,
      passiveWheel: false,
    });

    viewport.drag().pinch().decelerate();
    setTimeout(() => viewport.wheel(), 0);

    const fitScale = Math.min(
      screenWidth / imageWidth,
      screenHeight / imageHeight
    );

    viewport.clampZoom({ minScale: fitScale, maxScale: 1 });
    viewport.setZoom(fitScale, true);
    viewport.moveCenter(imageWidth / 2, imageHeight / 2);

    viewportRef.current = viewport;
    app.stage.addChild(viewport);

    const updateClamp = () => {
      const scale = viewport.scale.x;
      const halfWidth = screenWidth / (2 * scale);
      const halfHeight = screenHeight / (2 * scale);

      viewport.clamp({
        left: -halfWidth,
        right: imageWidth + halfWidth,
        top: -halfHeight,
        bottom: imageHeight + halfHeight,
      });
    };

    const updatePolygons = () => {
      let polygons = metadata.objects.filter(poly =>
        showQuarantined || !isQuarantined(poly.type, poly.stable_hash)
      );

      const typeColors = {
        stack: '#0000ff',
        target: '#0000ff',
        mosaic: '#ff0000',
      };
      const outputColor = '#00cc00';

      // Create polygon graphics and add them
      for (const poly of polygons) {
        const g = new PIXI.Graphics();
        g.zIndex = 9999;
        const color = poly.polygon_type === 'output' ? outputColor : (typeColors[poly.type] ?? '#ff0000');
        g.lineStyle(imageWidth / 500, color, poly.polygon_type === 'output' ? 0.6 : 1);
        // g.beginFill(poly.color ?? 0xff0000, 0.2);
        g.moveTo(poly.points[0][0], poly.points[0][1]);
        for (let i = 1; i < poly.points.length; i++) {
          g.lineTo(poly.points[i][0], poly.points[i][1]);
        }
        g.closePath();
        // g.endFill();

        // Interactive: click to navigate, hover to highlight.
        g.interactive = true;
        g.hitArea = new PIXI.Polygon(poly.points.flat());
        g.eventMode = 'static';
        g.cursor = 'pointer';
        g.on('pointertap', (event) => onClick(poly, event));
        // Add a transparent fill child that appears on hover.
        const highlight = new PIXI.Graphics();
        highlight.beginFill(0xffffff, 0.15);
        highlight.moveTo(poly.points[0][0], poly.points[0][1]);
        for (let i = 1; i < poly.points.length; i++) {
          highlight.lineTo(poly.points[i][0], poly.points[i][1]);
        }
        highlight.closePath();
        highlight.endFill();
        highlight.visible = false;
        g.addChild(highlight);
        g.on('pointerover', () => { highlight.visible = true; });
        g.on('pointerout', () => { highlight.visible = false; });

        viewport.addChild(g);

        // Per-panel overlays anchored to the top-right corner of the
        // polygon bounding box. Decision dot first (closest to the
        // corner), exposure clock to its left.
        if (poly.polygon_type === 'input') {
          const xs = poly.points.map(p => p[0]);
          const ys = poly.points.map(p => p[1]);
          const maxX = Math.max(...xs);
          const minY = Math.min(...ys);
          const dotRadius = imageWidth / 150;

          if (poly.has_decisions) {
            const dot = new PIXI.Graphics();
            dot.zIndex = 10000;
            dot.beginFill(0xffa726, 0.9); // amber
            dot.drawCircle(maxX, minY, dotRadius);
            dot.endFill();
            dot.lineStyle(dotRadius * 0.2, 0xffffff, 0.8);
            dot.drawCircle(maxX, minY, dotRadius);
            viewport.addChild(dot);
          }

          // Exposure-duration clock for the stack at the selected
          // filter. Input polygons are target-level (set by the backend
          // in metadata.py), so the active stack is referenced
          // explicitly via active_stack_hash.
          const stack = poly.active_stack_hash ? stacks?.[poly.active_stack_hash] : null;
          if (stack && typeof stack.integration === 'number') {
            const clockRadius = imageWidth / 80;
            const offset = poly.has_decisions ? dotRadius * 2.5 : 0;
            const clock = new PIXI.Graphics();
            clock.zIndex = 10000;
            drawExposureClock(
              clock,
              maxX - offset - clockRadius,
              minY + clockRadius,
              clockRadius,
              stack.integration / 3600,
            );
            viewport.addChild(clock);
          }
        }
      }

      viewport.sortChildren();
    }

    const updateTiles = () => {
      const scale = viewport.scale.x;
      const pixelsPerScreenPixel = 1 / scale;
      let bestZ = Math.ceil(maxZoom - Math.log2(pixelsPerScreenPixel));
      bestZ = Math.max(0, Math.min(maxZoom, bestZ));

      const levelScale = Math.pow(2, maxZoom - bestZ);
      const scaledWidth = Math.ceil(imageWidth / levelScale);
      const scaledHeight = Math.ceil(imageHeight / levelScale);

      const tilesX = Math.ceil(scaledWidth / TILE_SIZE);
      const tilesY = Math.ceil(scaledHeight / TILE_SIZE);

      const bounds = viewport.getVisibleBounds();
      const xStart = Math.floor(bounds.x / (TILE_SIZE * levelScale));
      const yStart = Math.floor(bounds.y / (TILE_SIZE * levelScale));
      const xEnd = Math.ceil((bounds.x + bounds.width) / (TILE_SIZE * levelScale));
      const yEnd = Math.ceil((bounds.y + bounds.height) / (TILE_SIZE * levelScale));

      for (let x = xStart; x < xEnd; x++) {
        for (let y = yStart; y < yEnd; y++) {
          if (x < 0 || y < 0 || x >= tilesX || y >= tilesY) continue;
          const key = `${bestZ}/${x}/${y}`;
          if (spritesRef.current[key]) continue;

          const tileLevel = qualityLevel || 'auto';
          const relativePath = `/tile/${objectType}/${imageHash}/${filter}/${tileLevel}/${bestZ}/${x}/${y}.png`;
          const url = new URL(relativePath, `${window.location.protocol}//${window.location.hostname}:${port}`);
          if (mtime) url.searchParams.set('v', mtime);
          PIXI.Texture.fromURL(url.href)
            .then((texture) => {
              const sprite = new PIXI.Sprite(texture);
              sprite.x = x * TILE_SIZE * levelScale;
              sprite.y = y * TILE_SIZE * levelScale;
              sprite.width = TILE_SIZE * levelScale;
              sprite.height = TILE_SIZE * levelScale;
              sprite.zIndex = bestZ;
              viewport.addChild(sprite);
              viewport.sortChildren();
              spritesRef.current[key] = sprite;
            })
            .catch((err) => {
              console.warn(`Failed to load tile: ${url}`, err);
            });
        }
      }

      updateClamp();
    };

    const updateAll = () => {
      updateTiles();

      if (showPolygons) {
        updatePolygons();
      }
    }

    updateAll();
    viewport.on('moved', updateAll);
    viewport.on('zoomed', updateAll);
    viewport.on('moved-end', updateAll);
    viewport.on('zoomed-end', updateAll);

    return () => {
      viewport.off('moved', updateAll);
      viewport.off('zoomed', updateAll);
      viewport.off('moved-end', updateAll);
      viewport.off('zoomed-end', updateAll);
      if (viewport.parent) {
        viewport.parent.removeChild(viewport);
      }
      try { viewport.destroy({ children: true }); } catch {}
      spritesRef.current = {};
    };
  }, [app, mosaics, imageHash, filter, imageWidth, imageHeight, maxZoom, worldWidth, worldHeight, showPolygons, stacks]);

  return null;
};

export default TileViewer;
