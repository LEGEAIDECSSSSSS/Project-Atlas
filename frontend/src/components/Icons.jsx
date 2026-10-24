const base = {
  width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
  strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true,
};
const make = (d) => function Icon(props) {
  return <svg {...base} {...props}>{d}</svg>;
};

export const IconPlus = make(<path d="M12 5v14M5 12h14" />);
export const IconSend = make(<path d="M12 19V5M5 12l7-7 7 7" />);
export const IconStop = make(<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />);
export const IconClip = make(<path d="m21 12-8.5 8.5a5 5 0 0 1-7-7L14 5a3.5 3.5 0 0 1 5 5l-8.5 8.5a2 2 0 0 1-3-3L15 7" />);
export const IconCopy = make(<><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></>);
export const IconCheck = make(<path d="m5 12 5 5 9-10" />);
export const IconDownload = make(<path d="M12 4v11m-5-5 5 5 5-5M5 20h14" />);
export const IconTrash = make(<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />);
export const IconMenu = make(<path d="M4 6h16M4 12h16M4 18h16" />);
export const IconRefresh = make(<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" />);
export const IconSun = make(<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>);
export const IconMoon = make(<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z" />);
export const IconX = make(<path d="M6 6l12 12M18 6 6 18" />);
export const IconFile = make(<><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>);
export const IconFolder = make(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />);
export const IconWrench = make(<path d="M14.7 6.3a4 4 0 0 0-5 5L3 18l3 3 6.7-6.7a4 4 0 0 0 5-5l-2.4 2.4-2.6-.6-.6-2.6z" />);
