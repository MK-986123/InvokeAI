import { Badge, Button, Flex, IconButton } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold, PiCheckBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  handleInstall: () => void;
  isInstalled: boolean;
  handleSelectModel?: () => void;
};

export const ModelResultItemActions = memo(({ handleInstall, isInstalled, handleSelectModel }: Props) => {
  const { t } = useTranslation();

  return (
    <Flex gap={2} shrink={0} pt={1} alignItems="center">
      {isInstalled ? (
        <>
          <Badge
            variant="subtle"
            colorScheme="green"
            display="flex"
            gap={1}
            alignItems="center"
            borderRadius="base"
            h="24px"
            cursor={handleSelectModel ? 'pointer' : 'default'}
            onClick={handleSelectModel}
          >
            <PiCheckBold size="14px" />
          </Badge>
          {handleSelectModel && (
            <IconButton
              aria-label={t('common.view')}
              tooltip={t('common.view')}
              icon={<PiArrowSquareOutBold size="14px" />}
              onClick={handleSelectModel}
              size="sm"
              variant="ghost"
            />
          )}
        </>
      ) : (
        <Button
          onClick={handleInstall}
          rightIcon={<PiPlusBold size="14px" />}
          textTransform="uppercase"
          letterSpacing="wider"
          fontSize="9px"
          size="sm"
        >
          {t('modelManager.install')}
        </Button>
      )}
    </Flex>
  );
});

ModelResultItemActions.displayName = 'ModelResultItemActions';
