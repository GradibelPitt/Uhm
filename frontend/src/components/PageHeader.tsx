export function PageHeader() {
  return <>
    <header className="topbar">
      <a className="brand" href="/" aria-label="Universal Haptic Model home"><span className="brand-mark">u<span>h</span>m<span className="brand-dot">.</span></span><span className="brand-name">UNIVERSAL<br />HAPTIC MODEL</span></a>
      <div className="topbar-right"><span className="status-dot" />RESEARCH WORKBENCH<span className="version-tag">V0.1</span></div>
    </header>
    <section className="intro">
      <div><p className="eyebrow">VISUAL PERCEPTION → EMBODIED REPRESENTATION</p><h1>See the impact. <span>Locate the experience.</span></h1><p className="intro-copy">Observe an event, identify its recipient, and map what can be known to the body.</p></div>
      <div className="scope-note"><span className="tiny-index">01 / CURRENT SCOPE</span><strong>Visual impact localization</strong><span>Video only · Device independent</span><span className="scope-rule">Unknown is a valid prediction.</span></div>
    </section>
  </>
}
