const divmod = (y, x) => {
  let div = Math.trunc(y / x);
  let rem = y % x;

  return [div, rem];
}

const ra_to_hms = (ra) => {
  let [hours, hours_remainder] = divmod(ra * 3600, 15 * 3600);
  let [minutes, minutes_remainder] = divmod(hours_remainder, 15 * 60);
  let seconds = Math.round(minutes_remainder / 15);

  return `${String(hours).padStart(2, '0')}h ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`;
}

const dec_to_dms = (dec) => {
  let negative = dec < 0 ? '-' : '';
  dec = Math.abs(dec);
  let [degrees, degrees_remainder] = divmod(dec * 3600, 3600);
  degrees_remainder = Math.abs(degrees_remainder);
  let [minutes, seconds] = divmod(degrees_remainder, 60);
  seconds = Math.round(seconds);

  return `${negative}${String(degrees).padStart(2, '0')}º ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`
}

function format_duration(seconds) {
  const units = [
    { label: 'd', value: 86400 },
    { label: 'h', value: 3600 },
    { label: 'm', value: 60 },
    { label: 's', value: 1 },
  ];

  return units
    .map(({ label, value }) => {
      const amount = Math.floor(seconds / value);
      seconds %= value;
      return amount > 0 ? `${amount}${label}` : null;
    })
    .filter(Boolean)
    .join(' ');
}


export { ra_to_hms, dec_to_dms, format_duration };
