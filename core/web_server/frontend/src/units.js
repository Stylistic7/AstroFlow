// Round to whole seconds FIRST, then split, so 59.5 s carries into the
// minute instead of printing as 60" (3440b517 read `00hr 48' 60"`, 2026-10-27).
const ra_to_hms = (ra) => {
  const total = Math.round(ra * 240) % 86400;  // seconds of time; 1° = 240 s
  const hours = Math.trunc(total / 3600);
  const minutes = Math.trunc((total % 3600) / 60);
  const seconds = total % 60;

  return `${String(hours).padStart(2, '0')}hr ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`;
}

const dec_to_dms = (dec) => {
  const negative = dec < 0 ? '-' : '';
  const total = Math.round(Math.abs(dec) * 3600);  // arcseconds
  const degrees = Math.trunc(total / 3600);
  const minutes = Math.trunc((total % 3600) / 60);
  const seconds = total % 60;

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
