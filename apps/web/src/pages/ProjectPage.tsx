import { useParams } from 'react-router-dom';

export function ProjectPage() {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <div>
      <h1>Project Overview</h1>
      <p>Project ID: {projectId}</p>
    </div>
  );
}
