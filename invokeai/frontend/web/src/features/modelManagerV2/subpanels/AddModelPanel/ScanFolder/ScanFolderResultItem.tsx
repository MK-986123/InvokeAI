import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { ModelResultItemActions } from 'features/modelManagerV2/subpanels/AddModelPanel/ModelResultItemActions';
import { memo, useCallback, useMemo } from 'react';
import type { ScanFolderResponse } from 'services/api/endpoints/models';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';

type Props = {
  result: ScanFolderResponse[number];
  installModel: (source: string) => void;
};

const scanFolderResultItemSx: SystemStyleObject = {
  alignItems: 'center',
  justifyContent: 'space-between',
  w: '100%',
  py: 2,
  px: 1,
  gap: 3,
  borderBottomWidth: '1px',
  borderColor: 'base.700',
};

export const ScanModelResultItem = memo(({ result, installModel }: Props) => {
  const dispatch = useAppDispatch();
  const { modelList } = useGetModelConfigsQuery(undefined, {
    selectFromResult: ({ data }) => ({ modelList: data ? modelConfigsAdapterSelectors.selectAll(data) : EMPTY_ARRAY }),
  });

  const handleInstall = useCallback(() => {
    installModel(result.path);
  }, [installModel, result]);

  const installedModel = useMemo(() => {
    if (!result.is_installed) {
      return null;
    }
    return modelList.find((mc) => mc.path === result.path || mc.source === result.path);
  }, [result.is_installed, result.path, modelList]);

  const handleSelectModel = useCallback(() => {
    if (installedModel) {
      dispatch(setSelectedModelKey(installedModel.key));
    }
  }, [dispatch, installedModel]);

  const modelDisplayName = useMemo(() => {
    const normalizedPath = result.path.replace(/\\/g, '/').replace(/\/+$/, '');

    // Extract filename/folder name from path
    const lastSlashIndex = normalizedPath.lastIndexOf('/');
    return lastSlashIndex === -1 ? normalizedPath : normalizedPath.slice(lastSlashIndex + 1);
  }, [result.path]);

  const modelPathParts = result.path.split(/[/\\]/);

  return (
    <Flex sx={scanFolderResultItemSx}>
      <Flex fontSize="sm" flexDir="column">
        {/* Model Title */}
        <Text fontWeight="semibold">{modelDisplayName}</Text>
        {/* Model Path */}
        <Flex flexWrap="wrap" color="base.200">
          {modelPathParts.map((part, index) => (
            <Text key={index} variant="subtext">
              {part}
              {index < modelPathParts.length - 1 && '/'}
            </Text>
          ))}
        </Flex>
      </Flex>
      <ModelResultItemActions
        handleInstall={handleInstall}
        isInstalled={result.is_installed}
        handleSelectModel={installedModel ? handleSelectModel : undefined}
      />
    </Flex>
  );
});

ScanModelResultItem.displayName = 'ScanModelResultItem';
