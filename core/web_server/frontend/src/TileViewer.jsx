import { useEffect, useState } from 'react';
import { Texture } from 'pixi.js';
import { Stage, Container, Sprite } from '@pixi/react';
import { useParams } from 'react-router-dom';

const TILE_SIZE = 256;

const TileViewer = ({ maxZoom = 5, minZoom = 0 }) => {
  const { imageHash } = useParams();

  if (!imageHash) return null;

  return (
    <Stage width={800} height={600} options={{ background: 0x000000 }}>
      <Container>
        <TileLayer imageHash={imageHash} maxZoom={maxZoom} minZoom={minZoom} />
      </Container>
    </Stage>
  );
};

const TileLayer = ({ imageHash, maxZoom, minZoom }) => {
  const [tiles, setTiles] = useState([]);

  useEffect(() => {
    const z = maxZoom;
    const xStart = 0;
    const yStart = 0;
    const xEnd = Math.ceil(800 / TILE_SIZE);
    const yEnd = Math.ceil(600 / TILE_SIZE);

    const newTiles = [];

    for (let x = xStart; x < xEnd; x++) {
      for (let y = yStart; y < yEnd; y++) {
        const url = `/tiles/${imageHash}/${z}/${x}/${y}.png`;
        newTiles.push({ url, x, y });
      }
    }

    setTiles(newTiles);
  }, [imageHash, maxZoom]);

  return (
    <>
      {tiles.map(({ url, x, y }) => (
        <SpriteTile key={url} url={url} x={x * TILE_SIZE} y={y * TILE_SIZE} />
      ))}
    </>
  );
};

const SpriteTile = ({ url, x, y }) => {
  const [texture, setTexture] = useState(null);

  useEffect(() => {
    let isMounted = true;
    Texture.fromURL(url)
      .then((tex) => {
        if (isMounted) setTexture(tex);
      })
      .catch(() => { });
    return () => {
      isMounted = false;
    };
  }, [url]);

  if (!texture) return null;
  return <Sprite texture={texture} x={x} y={y} width={TILE_SIZE} height={TILE_SIZE} />;
};

export default TileViewer;