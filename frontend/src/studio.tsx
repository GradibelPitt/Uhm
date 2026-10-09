import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ContactWorkbench } from './ContactWorkbench'
import './studio.css'

createRoot(document.getElementById('root')!).render(<StrictMode><ContactWorkbench /></StrictMode>)
