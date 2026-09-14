// @vitest-environment happy-dom
import { act } from 'react';
import type { Root } from 'react-dom/client';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelResultItemActions } from './ModelResultItemActions';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (str: string) => str,
  }),
}));

describe('ModelResultItemActions', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it('renders install button when not installed', () => {
    const handleInstall = vi.fn();

    act(() => {
      root.render(<ModelResultItemActions isInstalled={false} handleInstall={handleInstall} />);
    });

    const button = container.querySelector('button');
    expect(button).not.toBeNull();
    expect(button?.textContent).contains('modelManager.install');

    act(() => {
      button?.click();
    });
    expect(handleInstall).toHaveBeenCalledTimes(1);
  });

  it('renders installed badge and view action button when installed and handleSelectModel is provided', () => {
    const handleInstall = vi.fn();
    const handleSelectModel = vi.fn();

    act(() => {
      root.render(
        <ModelResultItemActions
          isInstalled={true}
          handleInstall={handleInstall}
          handleSelectModel={handleSelectModel}
        />
      );
    });

    const buttons = container.querySelectorAll('button');
    const viewButton = Array.from(buttons).find((b) => b.getAttribute('aria-label') === 'common.view');
    expect(viewButton).toBeDefined();

    act(() => {
      viewButton?.click();
    });
    expect(handleSelectModel).toHaveBeenCalledTimes(1);
  });

  it('renders installed badge without view button when handleSelectModel is not provided', () => {
    const handleInstall = vi.fn();

    act(() => {
      root.render(<ModelResultItemActions isInstalled={true} handleInstall={handleInstall} />);
    });

    const viewButton = container.querySelector('button[aria-label="common.view"]');
    expect(viewButton).toBeNull();
  });
});
