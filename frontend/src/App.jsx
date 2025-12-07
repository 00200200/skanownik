import { useRef, useState } from 'react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
	const fileInputRef = useRef(null);
	const [files, setFiles] = useState([]);
	const [loading, setLoading] = useState(false);
	const [results, setResults] = useState([]);
	const [error, setError] = useState('');
	const [useFolder, setUseFolder] = useState(false);

	const exportData = results.map(({ image, ...rest }) => rest);
    const jsonString = JSON.stringify(exportData, null, 2);

	const onSelect = e => {
		const list = Array.from(e.target.files || []);
		setFiles(list);
		setResults([]);
		setError('');
	};

	const analyze = async () => {
		if (!files.length) {
			setError('Wybierz plik lub folder.');
			return;
		}
		setLoading(true);
		setError('');
		const newResults = [];
		for (const file of files) {
			const formData = new FormData();
			formData.append('file', file);
			try {
				const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: formData });
				const data = await res.json();

				if (data.skipped) continue;

				if (!res.ok || data.error) throw new Error(data.error || 'Błąd analizy');
				newResults.push({
					name: file.webkitRelativePath || file.name,
					vehicle: data.vehicle_type,
					score: data.anomaly_score,
					is_anomaly: data.is_anomaly,
                    anomalies: data.anomalies,
					image: data.image_base64,
				});
			} catch (err) {
				newResults.push({
					name: file.webkitRelativePath || file.name,
					error: err.message || 'Błąd',
				});
			}
		}
		setResults(newResults);
		setLoading(false);
	};

    const downloadJSON = () => {
        if (!results.length) return;
        // Remove image data to keep JSON small
        
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'skanownik_results.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

	return (
		<div className='min-h-screen bg-slate-50 text-slate-900'>
			<div className='mx-auto max-w-5xl px-4 py-10'>
				<header className='mb-8 text-center'>
					<h1 className='text-3xl font-bold'>Skanownik</h1>
					<p className='mt-2 text-slate-600'>Wyślij pojedynczy plik lub cały folder do analizy.</p>
				</header>

				<section className='rounded-2xl bg-white p-6 shadow-sm border border-slate-200'>
					<div className='flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between'>
						<div className='space-y-1'>
							<p className='text-sm font-medium text-slate-700'>Wejście</p>
							<label className='inline-flex items-center gap-2 text-sm text-slate-600'>
								<input type='checkbox' checked={useFolder} onChange={e => setUseFolder(e.target.checked)} />
								Tryb folderu (wiele plików)
							</label>
						</div>
						<div className='flex flex-col gap-2 sm:flex-row sm:items-center'>
							<label className='inline-flex cursor-pointer items-center justify-center rounded-lg border border-dashed border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:border-blue-500 hover:text-blue-600'>
								<input
									ref={fileInputRef}
									type='file'
									className='hidden'
									accept='.bmp'
									multiple
									{...(useFolder ? { directory: '', webkitdirectory: '' } : {})}
									onChange={onSelect}
								/>
								Wybierz {useFolder ? 'folder' : 'plik'}
							</label>
							<button
								type='button'
								onClick={analyze}
								disabled={loading || !files.length}
								className='inline-flex items-center justify-center rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300'>
								{loading ? 'Analizuję...' : 'Wyślij do analizy'}
							</button>
							
                            {results.length > 0 && (
                                <button
                                    type='button'
                                    onClick={downloadJSON}
                                    className='inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-5 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50'>
                                    Pobierz JSON
                                </button>
                            )}
						</div>
					</div>
                    <p className='mt-2 text-xs text-slate-500 text-right'>Obsługiwane pliki: <b>.czarno.bmp</b></p>

					{error && <p className='mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700'>{error}</p>}

					{files.length > 0 && (
						<p className='mt-3 text-sm text-slate-600'>
							Wybrano: {files.length} plik(ów). Pierwszy: {files[0].webkitRelativePath || files[0].name}
						</p>
					)}
				</section>

				<section className='mt-6 space-y-4'>
					{loading && (
						<div className='rounded-xl border border-slate-200 bg-white p-4 shadow-sm text-sm text-slate-700'>
							Trwa przetwarzanie...
						</div>
					)}

					{results.map(r => (
						<div key={r.name} className='rounded-xl border border-slate-200 bg-white p-4 shadow-sm flex flex-col gap-3'>
							<div className='flex items-center justify-between'>
								<div>
									<p className='text-sm font-semibold text-slate-800'>{r.name}</p>
									{r.error ? (
										<p className='text-sm text-red-600'>{r.error}</p>
									) : (
										<div className='text-sm text-slate-600'>
											<p>{r.vehicle?.toUpperCase?.() || '-'} • wynik {r.score?.toFixed?.(4) ?? '-'}</p>
                                            {r.anomalies && r.anomalies.length > 0 && (
                                                <p className='text-xs text-slate-500 mt-1'>
                                                    Wykryto {r.anomalies.length} obszar(ów) anomalii
                                                </p>
                                            )}
										</div>
									)}
								</div>
								{!r.error && (
									<span
										className={`rounded-full px-3 py-1 text-xs font-semibold ${
											r.is_anomaly
												? 'bg-red-100 text-red-700 border border-red-200'
												: 'bg-green-100 text-green-700 border border-green-200'
										}`}>
										{r.is_anomaly ? 'ANOMALIA' : 'OK'}
									</span>
								)}
							</div>
							{r.image && (
								<img
									className='max-h-80 w-full rounded-lg border border-slate-200 object-contain bg-slate-50'
									src={`data:image/jpeg;base64,${r.image}`}
									alt={r.name}
								/>
							)}
						</div>
					))}
				</section>

				{jsonString !== "[]" ? (
					<section className='p-3 mt-2 bg-slate-100 rounded-lg'>
						<pre>
							{jsonString}
						</pre>
					</section>
				) : null}
			</div>
		</div>
	);
}

export default App;
