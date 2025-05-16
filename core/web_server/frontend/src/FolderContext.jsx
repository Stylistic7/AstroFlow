import { createContext, useContext, useState } from 'react';

const FolderContext = createContext(null);

export function FolderProvider({ children }) {
    const [lastFolder, setLastFolder] = useState('/Users');
    return (
        <FolderContext.Provider value={{ lastFolder, setLastFolder }}>
            {children}
        </FolderContext.Provider>
    );
}

export function useFolder() {
    const ctx = useContext(FolderContext);
    if (!ctx) throw new Error('useFolder must be used within a <FolderProvider>');
    return ctx;
}
