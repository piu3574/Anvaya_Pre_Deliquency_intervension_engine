interface PlaceholderProps {
  title: string;
  subtitle: string;
}

export default function PlaceholderPage({ title, subtitle }: PlaceholderProps) {
  return (
    <section className="placeholder-view">
      <p className="eyebrow">Module In Progress</p>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </section>
  );
}
