import Typography from '@mui/material/Typography';

export default function ColorChannelSelector({ stacks, targets, mosaics, mosaic_id, filter, setFilter }) {
  const mosaic = mosaics[mosaic_id];

  if (!mosaic) {
    return <Typography>Mosaic not found: {mosaic_id}</Typography>;
  }

  const colorForFilter = (filter) => {
    let rgb = {
      'L': [51, 51, 51],
      'R': [230, 0, 0],
      'G': [0, 204, 0],
      'B': [0, 0, 204],
      'H': [159, 43, 104],
      'S': [140, 0, 0],
      'O': [0, 204, 204],
    }[filter];

    return 'rgb(' + rgb.join(', ') + ')'
  };

  const getFilters = () => {
    let filter_array = [];

    for (let filter of ['H', 'S', 'O', 'L', 'R', 'G', 'B']) {
      let found = false;
      for (let target_id of mosaic.target_ids) {
        for (let stack_id of targets[target_id].stack_ids) {
          if (stacks[stack_id].filter === filter) {
            found = true;
          }
        }
      }

      if (found) {
        filter_array.push(filter);
      }
    }

    return filter_array;
  }

  const filters = getFilters();

  const imageType = "mosaic_thumbnail_image_for_filter";

  return (
    <div
      style={{
        // width: "100%",
        display: "flex",
        justifyContent: "center"
      }}
    >
      {filters.map(f => {
        const color = colorForFilter(f);
        return <div
          key={f}
          style={{
            float: "left",
            margin: "0.5em",
            padding: "0.5em",
            border: f === filter ? `1px solid ${color}` : "",
          }}
        >
          <span
            style={{
              color: color,
              margin: "0.5em"
            }}
            onClick={event => setFilter(f)}
          >
            {f}
          </span>
        </div>
      })}
    </div >
  );
}
