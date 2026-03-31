import { useEffect, useRef, useState, useLayoutEffect, useContext } from 'react';
import * as PIXI from 'pixi.js';
import { Stage, useApp } from '@pixi/react';
import { useParams, useNavigate } from 'react-router-dom';
import { Viewport } from 'pixi-viewport';

import AstroFlowContext from './AstroFlowContext.jsx';


const TILE_SIZE = 256;

const TileViewer = ({ objectType: objectTypeProp, imageHash: imageHashProp, filter: filterProp, showPolygons }) => {
  const { objectType: objectTypeParam, imageHash: imageHashParam, filter: filterParam } = useParams();

  const objectType = objectTypeProp ?? objectTypeParam;
  const imageHash = imageHashProp ?? imageHashParam;
  const filter = filterProp ?? filterParam;

  const [metadata, setMetadata] = useState(null);
  const [tileMetadata, setTileMetadata] = useState(null);
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  const { mosaics, stacks, targets } = useContext(AstroFlowContext);

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
    const url = filter !== undefined
      ? `/api/metadata/${objectType}/${imageHash}/${filter}`
      : `/api/metadata/${objectType}/${imageHash}`;

    fetch(url)
      .then((res) => res.json())
      .then((meta) => setMetadata(meta));
  }, [objectType, imageHash, filter]);

  useEffect(() => {
    fetch(`/tile_metadata/${objectType}/${imageHash}/${filter}`)
      .then((res) => res.json())
      .then((meta) => setTileMetadata(meta));
  }, [objectType, imageHash, filter]);

  const onClick = (poly, event) => {
    const e = event.nativeEvent; // underlying DOM PointerEvent

    // Navigate to the stack for the current filter, if available.
    let url = `${poly.type}/${poly.stable_hash}`;
    if (poly.type === 'target' && filter && targets?.[poly.stable_hash]) {
      const target = targets[poly.stable_hash];
      const stackId = target.stack_ids?.find(s => stacks?.[s]?.filter === filter);
      if (stackId) {
        url += `/stack/${stackId}`;
      }
    }

    if (e.metaKey || e.ctrlKey) {
      window.open(url, '_blank');
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
          port={tileMetadata.port}
          imageWidth={tileMetadata.width}
          imageHeight={tileMetadata.height}
          worldWidth={worldWidth}
          worldHeight={worldHeight}
          maxZoom={tileMetadata.maxZoom}
          showPolygons={showPolygons}
          metadata={metadata}
          mosaics={mosaics}
          onClick={onClick}
        />
      </Stage>
    </div>
  );
};

const ImperativeViewport = ({ objectType, imageHash, filter, port, imageWidth, imageHeight, worldWidth, worldHeight, maxZoom, showPolygons, metadata, mosaics, onClick }) => {
  const app = useApp();
  const viewportRef = useRef(null);
  const spritesRef = useRef({});

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
      let polygons = metadata.objects;

      const typeColors = {
        stack: '#0000ff',
        target: '#0000ff',
        mosaic: '#ff0000',
      };

      // Create polygon graphics and add them
      for (const poly of polygons) {
        const g = new PIXI.Graphics();
        g.zIndex = 9999;
        g.lineStyle(imageWidth / 500, typeColors[poly.type], 1);
        // g.beginFill(poly.color ?? 0xff0000, 0.2);
        g.moveTo(poly.points[0][0], poly.points[0][1]);
        for (let i = 1; i < poly.points.length; i++) {
          g.lineTo(poly.points[i][0], poly.points[i][1]);
        }
        g.closePath();
        // g.endFill();

        // Optional: make interactive
        g.interactive = true;
        g.hitArea = new PIXI.Polygon(poly.points.flat()); // hit test against polygon area
        g.eventMode = 'static';
        g.cursor = 'pointer';
        g.on('pointertap', (event) => onClick(poly, event));

        viewport.addChild(g);

        // Decision indicator: amber dot at the top-right of the polygon bounding box.
        if (poly.has_decisions) {
          const xs = poly.points.map(p => p[0]);
          const ys = poly.points.map(p => p[1]);
          const maxX = Math.max(...xs);
          const minY = Math.min(...ys);
          const dotRadius = imageWidth / 150;

          const dot = new PIXI.Graphics();
          dot.zIndex = 10000;
          dot.beginFill(0xffa726, 0.9); // amber
          dot.drawCircle(maxX, minY, dotRadius);
          dot.endFill();
          // White outline for visibility
          dot.lineStyle(dotRadius * 0.2, 0xffffff, 0.8);
          dot.drawCircle(maxX, minY, dotRadius);
          viewport.addChild(dot);
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

          const relativePath = `/tile/${objectType}/${imageHash}/${filter}/${bestZ}/${x}/${y}.png`;
          const url = new URL(relativePath, `${window.location.protocol}//${window.location.hostname}:${port}`);
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
  }, [app, mosaics, imageHash, filter, imageWidth, imageHeight, maxZoom, worldWidth, worldHeight, showPolygons]);

  return null;
};

export default TileViewer;
